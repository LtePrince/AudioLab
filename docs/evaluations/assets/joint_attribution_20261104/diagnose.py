"""E24: fixed E23/e300 checkpoint, causal history/audio interventions, no training.

Calls the existing generator. Read-only wrappers record native decisions or
force a declared prefix; every token in the scored continuation is free.
Community references supply timing only, never invented 4K position labels.
"""
from pathlib import Path
import argparse
from collections import defaultdict, deque
import gc
import json
import math
import os
import sys
import time
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/joint_audio_ar.py').exists())
sys.path.insert(0, str(ROOT))
import torch

from script.train_joint_audio_a import mel_from_wave
from script.train_soft_onset import load_sample
from script.train_windowed_corpus import natural_reference, scoring_module
from src.anchored_ar import MIN_PROGRESS_SECONDS
from src.audio_understanding import SAMPLE_RATE
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import BOS, ChartTokenizer
from src.joint_audio_ar import JointAudioWindowAR
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import generate, window_intervals

ASSET = Path(__file__).resolve().parent
OUT = ROOT/'out/joint_attribution_20261104'
RUN = ROOT/'out/joint_audio_scratch_20261102'
MEL = ROOT/'out/joint_audio_a_20261102/mel'
SELECT = ROOT/'docs/evaluations/assets/phira_generalization10_20261029/selection.json'
FOCUS = {'Eltaw_IN': [6, 13], 'Incyde_IN': [18, 19, 20],
         '40717': [17, 18], '52417': [18, 19, 20, 21, 22, 23, 24, 25, 26]}


def read(path):
    return json.loads(Path(path).read_text())


def reference_trace(target):
    """Exact physical target times, avoiding float32 frame->seconds round trips."""
    tk = ChartTokenizer(4)
    times = dict(zip(target['decisions'].tolist(), target['next_seconds'].tolist()))
    clock = 0.
    result = []
    for i, token in enumerate(target['tokens'].tolist()):
        start = float(target['bounds'][i, 0]); window = round(start/10.)
        if token == tk.cond(2):
            clock = start
        elif token == tk.cond(0):
            clock = times[i-1]
        result.append([token, clock, window])
    return result


def prefix_before_window(trace, window):
    tk = ChartTokenizer(4)
    indices = [i for i, (token, _, w) in enumerate(trace)
               if w == window and token in (BOS, tk.cond(2))]
    if len(indices) != 1:
        raise ValueError('one boundary input required')
    # Include the boundary INPUT, but do not force its predicted output.
    prefix = trace[:indices[0]+1]
    assert all(t < 10*window for _, t, _ in prefix[:-1])
    assert prefix[-1][1] == 10*window
    return prefix


@torch.inference_mode()
def rollout(model, condition, bpm, budget, *, prefix=None, timing=None,
            short_history_window=None, zero_summary_window=None):
    chart = model.chart; decoder = chart.decoder; tk = ChartTokenizer(4)
    anchor, end_group, end_window = (tk.cond(i) for i in range(3))
    original_step = decoder.step; original_frame = chart.frame_logits
    original_fraction = chart.fraction
    state = dict(window=0, clock=0., token=BOS, pos=0)
    timing_groups = defaultdict(list)
    for t in timing or []:
        timing_groups[int(t//10)].append(t)
    counters = defaultdict(int); decisions = []; suffix = deque(maxlen=64)
    forced_layout = 0; forced_decisions = 0

    def step(token, frame, pos, caches, length, valid, history=None, input_condition=None):
        nonlocal forced_layout
        value = int(token[0])
        if value == end_window:
            state['window'] += 1; state['clock'] = 10.*state['window']
        state.update(token=value, pos=pos, force=None, pending_time=None)
        w = state['window']
        if prefix is not None and pos < len(prefix):
            expected, seconds, expected_window = prefix[pos]
            assert value == expected and w == expected_window
            assert abs(state['clock']-seconds) < 1e-5, (pos, state['clock'], seconds)
            if pos+1 < len(prefix):
                state['force'] = prefix[pos+1]
        if value in (BOS, end_window) and w == short_history_window:
            # Recompute only the self-attention suffix. Original causal summaries,
            # absolute clocks, RoPE positions and all audio remain identical.
            for cache in caches:
                cache['k'] = cache['v'] = None
            for saved in suffix:
                original_step(*saved[:3], caches, length, valid,
                              history=saved[3], input_condition=saved[4])
        if value in (BOS, end_window) and w == zero_summary_window:
            history = torch.zeros_like(history)
        logits, hidden = original_step(token, frame, pos, caches, length, valid,
                                       history=history, input_condition=input_condition)
        suffix.append((token.clone(), frame.clone(), pos, history.clone(), input_condition.clone()))
        forced = state['force']
        if forced is not None and value not in (BOS, end_group, end_window):
            logits = torch.full_like(logits, -1e9); logits[0, forced[0]] = 0.
            forced_layout += 1
        return logits, hidden

    def stop_hook(module, inputs, output):
        nonlocal forced_decisions
        logit = float(output.reshape(-1)[0]); w = state['window']
        row = dict(position=state['pos'], window=w, clock=state['clock'],
                   window_start=state['token'] in (BOS, end_window),
                   native_stop_probability=float(output.sigmoid().reshape(-1)[0]),
                   native_end=logit > 0, forced=False)
        forced = state['force']; end = None
        if forced is not None:
            assert forced[0] in (anchor, end_window)
            end = forced[0] == end_window
            if not end:
                state['pending_time'] = forced[1]
        elif timing is not None:
            index = counters[w]; end = index == len(timing_groups[w])
            if not end:
                state['pending_time'] = timing_groups[w][index]; counters[w] += 1
        if end is not None:
            row.update(forced=True, forced_end=end); forced_decisions += 1
            output = torch.full_like(output, 40. if end else -40.)
        decisions.append(row)
        return output

    def frame_logits(hidden, memory, allowed):
        scores = original_frame(hidden, memory, allowed)
        native = int(scores.argmax(-1)[0]); state['selected'] = native
        pending = state['pending_time']
        if pending is not None:
            edges = torch.arange(memory.shape[1]+1, dtype=torch.float64, device=memory.device)*MEMORY_SECONDS
            chosen = int(torch.searchsorted(edges, edges.new_tensor(pending), right=True))-1
            assert bool(allowed[0, chosen]), (pending, state)
            scores = torch.full_like(scores, -1e9).masked_fill(~allowed, -torch.inf)
            scores[0, chosen] = 0.; state['selected'] = chosen
        decisions[-1]['native_pointer_frame'] = native
        return scores

    def fraction(hidden, selected_memory):
        result = original_fraction(hidden, selected_memory)
        lower = max(10.*state['window'], state['clock'] +
                    (MIN_PROGRESS_SECONDS if state['token'] == end_group else 0.))
        upper = min(10.*(state['window']+1), condition['duration'])
        index = state['selected']
        a, b = max(lower, index*MEMORY_SECONDS), min(upper, (index+1)*MEMORY_SECONDS)
        if state['pending_time'] is not None:
            native_time = min(a+float(result[0])*(b-a), math.nextafter(b, a))
            if native_time != state['pending_time']:
                result = result.new_tensor([(state['pending_time']-a)/(b-a)])
        state['clock'] = min(a+float(result[0])*(b-a), math.nextafter(b, a))
        decisions[-1]['selected_next_seconds'] = state['clock']
        return result

    decoder.step = step; chart.frame_logits = frame_logits; chart.fraction = fraction
    hook = chart.stop_head.register_forward_hook(stop_hook)
    try:
        prediction = generate(chart, condition, bpm, use_soft=False, greedy=True,
                              max_tokens=12000, check=budget.check)
    finally:
        decoder.step = original_step; chart.frame_logits = original_frame
        chart.fraction = original_fraction; hook.remove()
    actual = [r for r in prediction['trace'] if r[0] in (BOS, end_group, end_window)]
    assert len(actual) == len(decisions)
    for row, (_, seconds, window) in zip(decisions, actual, strict=True):
        assert abs(row['clock']-seconds) < 1e-8 and row['window'] == window
    return dict(prediction=prediction, decisions=decisions,
                forced_prefix_layout_steps=forced_layout, forced_prefix_decisions=forced_decisions)


def window_metrics(reference, prediction, index, duration):
    a, b = index*10., min((index+1)*10., duration)
    truth = sorted({n['time_seconds'] for n in reference if a <= n['time_seconds'] < b})
    notes = [n for n in prediction['notes'] if a <= n['time_seconds'] < b]
    times = sorted({n['time_seconds'] for n in notes})
    matched = len(match_events(truth, times, .05))
    record = next(w for w in prediction['windows'] if w['index'] == index)
    return dict(window=index, reference_onsets=len(truth), generated_onsets=len(times), notes=len(notes),
                matched=matched, recall=matched/len(truth) if truth else None,
                precision=matched/len(times) if times else None,
                f1=2*matched/(len(truth)+len(times)) if truth or times else None,
                termination=record['termination'], decision_clock=record['decision_clock'])


@torch.inference_mode()
def teacher_diagnostics(model, condition, target):
    chart = model.chart; dec = chart.decoder; device = next(model.parameters()).device
    memory = dec.prepare_memory(condition['features'])
    _, hidden = dec(target['tokens'][None].to(device), target['frames'][None].to(device),
        memory, condition['valid'], history=target['history'][None].to(device), return_hidden=True,
        input_condition=chart.window_condition(target['bounds'].to(device))[None])
    rows = []; trace = reference_trace(target)
    for j, i in enumerate(target['decisions'].tolist()):
        token, clock, window = trace[i]; h = hidden[0, i:i+1]
        p = float(chart.stop_head(h).sigmoid()[0, 0]); stop = bool(target['stop_targets'][j])
        rows.append(dict(position=i, window=window, clock=clock, truth_end=stop,
            stop_probability=p, predicted_end=p > .5,
            window_start=token in (BOS, ChartTokenizer(4).cond(2))))
    return dict(decisions=rows, continue_count=sum(not r['truth_end'] for r in rows),
        false_stops=sum(r['predicted_end'] and not r['truth_end'] for r in rows),
        stop_count=sum(r['truth_end'] for r in rows),
        missed_stops=sum(not r['predicted_end'] and r['truth_end'] for r in rows))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); os.chdir(ROOT); OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2); device = torch.device(args.device)
    budget = ResourceBudget(SimpleNamespace(min_free_disk_gib=8, min_available_ram_gib=8,
        max_rss_gib=6, gpu_memory_fraction=.65, keep_checkpoints=1), device, OUT)
    checkpoint = RUN/'model_latest.pt'; expected = read(ASSET.parent/'joint_audio_scratch_20261102/final_result.json')
    assert sha256(checkpoint) == expected['weights'][0]['sha256']
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert loaded['epoch'] == 300 and loaded['steps'] == 90000
    model = JointAudioWindowAR(loaded['config']['encoder_config'], loaded['config']['decoder_config']).to(device).eval()
    model.load_state_dict(loaded['model'], strict=True); del loaded
    manifest = read(MEL/'manifest.json'); selection = read(SELECT); cache = read('out/full_corpus_v1/cache/cache.json')
    processor = AudioGPUprocessor(device='cpu'); profile = read('benchmarks/v1/protocol.json')['rhythm']; scorer = scoring_module()
    plan = dict(experiment='E24', checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha256(checkpoint),
        epoch=300, steps=90000, training_updates=0, focus=FOCUS, source_sha256={
            path:sha256(path) for path in [str(Path(__file__).relative_to(ROOT)), 'src/windowed_ar.py',
                'src/joint_audio_ar.py', 'src/models/chart_decoder.py', 'src/models/chart_history.py']},
        controls=dict(greedy=True, seed=42, max_tokens=12000, full_audio=True, grid_snap=False),
        scope='Selected failure attribution, not population generalization. True 4K history only on training songs. '
              'Community timing-prefix uses reference onsets plus model-generated layouts. '
              'Any forced prefix is excluded from continuation scoring. Physical silence is a control, '
              'zero features an out-of-distribution sensitivity test. No training or product changes.')
    amendment = read(ASSET/'validation_amendment.json') if (ASSET/'validation_amendment.json').exists() else None
    if amendment is not None:
        assert sha256(ASSET/'plan_initial.json') == amendment['initial_plan_sha256']
        assert sha256(Path(__file__)) == amendment['revised_script_sha256']
        plan['validation_amendment'] = amendment
    plan_path = ASSET/'plan.json'
    if plan_path.exists():
        assert read(plan_path) == plan, 'diagnostic source/config changed; do not mix results'
    else:
        write_json(plan_path, plan)
    results = []
    for song_id, windows in FOCUS.items():
        path = OUT/f'{song_id}.json'
        if path.exists():
            record = read(path)
            compatible = {sha256(plan_path)}
            if amendment is not None and song_id in amendment['preserved_completed_songs']:
                compatible.add(amendment['initial_plan_sha256'])
            assert record['plan_sha256'] in compatible
            results.append(record); continue
        print(json.dumps(dict(stage='song', id=song_id)), flush=True); budget.check(force=True)
        old = read(RUN/f'evaluation-e0300/{song_id}.json')
        train = old['split'] == 'train15'; target = None
        if train:
            row = next(r for r in manifest['rows'] if r['id'] == song_id)
            assert sha256(MEL/row['file']) == row['sha256']
            view = torch.load(MEL/row['file'], map_location='cpu', weights_only=True)['views'][0]
            target = view['target']; reference_prefix = reference_trace(target)
            source = next(r for r in cache['rows'] if r['id'] == song_id)
            sample = load_sample(Path('out/full_corpus_v1/cache'), source)
            reference = natural_reference(sample); meta = sample['routes']['4k']['reference']
            clock = dict(bpm=meta['timebase_bpm'], beat_zero_seconds=meta['offset_seconds'], verified=True)
            audio = source['audio']['path']; audio_sha = source['audio']['sha256']; del sample
        else:
            row = next(r for r in manifest['development'] if r['id'] == song_id)
            assert sha256(MEL/row['file']) == row['sha256']
            view = torch.load(MEL/row['file'], map_location='cpu', weights_only=True)
            song = next(s for s in selection['songs'] if str(s['id']) == song_id)
            assert sha256(song['reference']) == song['reference_sha256']
            reference = read(song['reference'])['notes']; clock = song['clock']
            audio = song['audio']; audio_sha = song['audio_sha256']
        with torch.inference_mode():
            condition = model.condition(view['mel'], view['duration'])
            native = rollout(model, condition, clock['bpm'], budget)
            assert native['prediction']['notes'] == old['prediction']['notes'], 'native replay differs from e300'
            assert native['prediction']['windows'] == old['prediction']['windows']
            record = dict(id=song_id, split=old['split'], plan_sha256=sha256(plan_path),
                exact_replay=True, source_audio_sha256=audio_sha, native_decisions=native['decisions'],
                native_metrics=old['metrics'], arms=[], teacher=None, audio=[])
            control_window = 20 if song_id=='52417' else windows[0]
            identical_prefix = prefix_before_window(native['prediction']['trace'], control_window)
            control = rollout(model, condition, clock['bpm'], budget, prefix=identical_prefix)
            assert control['prediction']['notes'] == native['prediction']['notes'], 'prefix replay control differs'
            assert [w for w in control['prediction']['windows'] if w['index']>=control_window] == [
                w for w in native['prediction']['windows'] if w['index']>=control_window]
            record['forced_prefix_metadata_differences'] = [dict(native=a,replayed=b)
                for a,b in zip(native['prediction']['windows'],control['prediction']['windows'],strict=True) if a!=b]
            record['exact_native_prefix_control'] = True
            del control
            if target is not None:
                record['teacher'] = teacher_diagnostics(model, condition, target)
            else:
                times = sorted({n['time_seconds'] for n in reference if 0 <= n['time_seconds'] < view['duration']})
                oracle = rollout(model, condition, clock['bpm'], budget, timing=times)
                assert len(oracle['prediction']['onsets_seconds']) == len(times)
                assert max(abs(a-b) for a,b in zip(times,oracle['prediction']['onsets_seconds'])) < 1e-5
                reference_prefix = oracle['prediction']['trace']
                record['oracle_timing_decisions'] = oracle['decisions']
                write_json(OUT/f'{song_id}-oracle-timing.json', oracle)
            write_json(OUT/f'{song_id}-native.json', native)
            # Score only the first wholly free window after the declared prefix.
            # A single reference prefix release also tracks subsequent free windows.
            release_windows = [windows[0]] if train else ([19,20,24] if song_id=='52417' else [17])
            for release in release_windows:
                prefix = prefix_before_window(reference_prefix, release)
                result = rollout(model, condition, clock['bpm'], budget, prefix=prefix)
                compared = [w for w in windows if w >= release]
                arm = dict(kind='true_4k_prefix' if train else 'reference_time_model_layout_prefix',
                    release_window=release, prefix_tokens=len(prefix),
                    windows=[window_metrics(reference, result['prediction'], w, view['duration']) for w in compared],
                    decisions=[r for r in result['decisions'] if r['window'] >= release])
                record['arms'].append(arm); write_json(OUT/f'{song_id}-prefix-w{release}.json', result)
                print(json.dumps(dict(stage='prefix', id=song_id, release=release, windows=arm['windows'])), flush=True)
            release = 20 if song_id=='52417' else windows[0]
            for kind, options in [('short_self_history',dict(short_history_window=release)),
                                  ('zero_current_summary',dict(zero_summary_window=release))]:
                result = rollout(model, condition, clock['bpm'], budget, **options)
                arm = dict(kind=kind, release_window=release,
                    windows=[window_metrics(reference,result['prediction'],w,view['duration']) for w in windows if w>=release],
                    decisions=[r for r in result['decisions'] if r['window']>=release])
                record['arms'].append(arm); write_json(OUT/f'{song_id}-{kind}.json', result)
            # Physical full-song silence and local muting are freshly re-encoded.
            # Replay identical native tokens up to release; only its continuation is free.
            assert sha256(audio) == audio_sha
            wave = processor.load_from_path(audio); native_prefix=prefix_before_window(native['prediction']['trace'], release)
            original_mel, duration = mel_from_wave(processor,wave)
            assert duration == view['duration'] and torch.equal(original_mel,view['mel'])
            start, end = release*10., min((release+1)*10.,duration)
            segment = wave[...,round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)]
            record['audio_evidence'] = dict(start=start,end=end,rms_dbfs=float(20*segment.square().mean().sqrt().clamp_min(1e-12).log10()))
            for name in ['physical_silence','local_mute','zero_features']:
                if name == 'zero_features':
                    altered = dict(condition,features=torch.zeros_like(condition['features']))
                else:
                    changed = torch.zeros_like(wave) if name=='physical_silence' else wave.clone()
                    if name=='local_mute':changed[...,round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)]=0
                    mel, duration = mel_from_wave(processor,changed)
                    altered = model.condition(mel,duration); del mel,changed
                result = rollout(model,altered,clock['bpm'],budget,prefix=native_prefix)
                record['audio'].append(dict(kind=name,release_window=release,
                    window=window_metrics(reference,result['prediction'],release,duration),
                    first_decision=next(r for r in result['decisions'] if r['window']==release),
                    scope='fixed native prefix, intervention re-encodes/replays all audio-conditioned hidden states'))
                write_json(OUT/f'{song_id}-{name}.json',result); del altered,result
            record['native_windows']=[window_metrics(reference,native['prediction'],w,view['duration']) for w in windows]
            record['resources']=budget.log_state(); write_json(path,record); results.append(record)
            print(json.dumps(dict(stage='song_done',id=song_id,resources=record['resources'])),flush=True)
            del native,condition,view,wave,original_mel
        gc.collect(); torch.cuda.empty_cache()
    write_json(ASSET/'results.json',dict(experiment='E24',status='completed',training_updates=0,
        checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha256(plan_path),results=results,
        resources=budget.log_state(),scope=plan['scope']))
    print(json.dumps(dict(status='completed',songs=len(results),output=str(ASSET/'results.json'))),flush=True)


if __name__ == '__main__':
    main()
