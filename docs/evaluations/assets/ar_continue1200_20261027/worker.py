"""Unchanged E14/E18 training, with fixed-15 evaluation inside its RNG fork."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import gc
import hashlib
import json
import statistics
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
sys.path.insert(0, str(ROOT))
import torch
from script import train_anchored_ar as training
from src.anchored_ar import AnchoredAR, generate, residuals_by_note
from src.audio_understanding.evaluation import match_events
from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
from src.model_benchmark import rhythm_bonus
from src.pattern_diagnostics import KINDS, LANE_X, patterns
from src.product.artifacts import write_json

PLAN = ROOT/'docs/evaluations/assets/runtime_selection15_20261026/selection.json'
METRIC = ROOT/'docs/evaluations/assets/freex_layout_best_20261026/analyze.py'
CACHE = ROOT/'out/full_corpus_v1/cache'
tree = ast.parse(METRIC.read_text())
# Reuse the exact existing group matcher without running its plotting program.
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
    and n.name in ('groups', 'compare')], type_ignores=[]), str(METRIC), 'exec'))
original_evaluate = training.evaluate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(sample, mode):
    route = sample['routes'][mode]
    clock = route['reference']
    scale = 60/(32*clock['timebase_bpm'])
    offsets = residuals_by_note(route) if mode == 'freex' else None
    notes = [dict(time_seconds=n.tick*scale+clock['offset_seconds'],
        end_seconds=(n.tick+n.dur)*scale+clock['offset_seconds'], position=n.lane,
        type=KINDS[n.type], screen_x=bin_to_hit(n.lane, offsets[n.tick,n.lane])
        if offsets is not None else LANE_X[n.lane]) for n in
        ChartTokenizer(64 if mode == 'freex' else 4).decode_tokens(route['tokens'].tolist(), strict=True)]
    return notes, clock


def score(reference_notes, prediction, clock, profile):
    result, _, _ = compare(reference_notes, prediction['notes'])
    rt = sorted({n['time_seconds'] for n in reference_notes})
    pt = prediction['onsets_seconds']
    matched = match_events(rt, pt, .05)
    result.update(selection_joint_f1=result['joint_time_bin_type_f1']
        if prediction['status'] == 'completed' else 0.,
        matched_onset_mae_ms=1000*statistics.mean(abs(a-b) for a,b in matched) if matched else None,
        grid_bonus=rhythm_bonus(pt, dict(bpm=clock['timebase_bpm'],
            beat_zero_seconds=clock['offset_seconds'], source='canonical reference chart clock',
            verified=False), profile)['bonus_per_onset'],
        first_onset_delta_seconds=pt[0]-rt[0] if pt else None,
        last_onset_delta_seconds=pt[-1]-rt[-1] if pt else None,
        tail_10s_recall=sum(a >= rt[-1]-10 for a,b in matched)/sum(t >= rt[-1]-10 for t in rt))
    for name, notes in [('reference', reference_notes), ('prediction', prediction['notes'])]:
        events = [dict(n, id=i, tick=round(n['time_seconds'], 6)) for i,n in enumerate(notes)]
        pattern = patterns(events)
        result[name+'_patterns'] = {k: v for k,v in pattern.items() if k != 'instances'}
    return result


@torch.inference_mode()
def evaluate15(checkpoint, mode, variant, output, budget, device):
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    assert sha(CACHE/'cache.json') == plan['cache_sha256']
    assert sha(METRIC) == plan['metric_sha256']
    fingerprint = sha(checkpoint)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert ck['config']['variant'] == variant
    assert ck['config']['decoder_config'].get('n_lanes', 4) == (4 if mode == '4k' else 64)
    epoch = ck['epoch']
    model = AnchoredAR(ck['config']['decoder_config']).to(device).eval()
    model.load_state_dict(ck['model'], strict=True)
    del ck
    profile = json.loads((ROOT/'benchmarks/v1/protocol.json').read_text())['rhythm']
    rows = []
    for i, row in enumerate(plan['songs']):
        budget.check(force=True)
        sample_path = CACHE/row['file']
        assert sha(sample_path) == row['sha256']
        sample = torch.load(sample_path, map_location='cpu', weights_only=True)
        truth, clock = reference(sample, mode)
        view = training.condition(sample, mode)
        prediction = generate(model, view, clock['timebase_bpm'], use_soft=variant == 'soft',
                              greedy=True, seed=42, check=budget.check)
        prediction.pop('trace', None)
        metric = score(truth, prediction, clock, profile)
        item = dict(song=row['id'], mode=mode, variant=variant, epoch=epoch,
            checkpoint_sha256=fingerprint, audio_sha256=row['audio']['sha256'],
            reference=clock, metrics=metric, prediction=prediction)
        write_json(output/f'{i:02d}.json', item)
        rows.append(dict(song=row['id'], status=prediction['status'], error=prediction.get('error'), metrics=metric))
        print(json.dumps(dict(stage='fixed15', mode=mode, variant=variant, epoch=epoch,
            done=i+1, song=row['id'], status=prediction['status'], joint_f1=metric['joint_time_bin_type_f1']),
            ensure_ascii=False), flush=True)
        del sample, view, truth, prediction, item
    summary = dict(schema='audiolab-continued-fixed15-v1', mode=mode, variant=variant,
        epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=fingerprint,
        selection_sha256=sha(PLAN), metric_sha256=sha(METRIC), greedy=True,
        scope='15 fixed training songs; no held-out generalization; no snapping or repairs',
        metric_note='Primary mean joint F1 counts unfinished output as zero. Raw metrics also retain failed prefixes.',
        completed=sum(r['status']=='completed' for r in rows), total=len(rows),
        mean_joint_f1=statistics.mean(r['metrics']['selection_joint_f1'] for r in rows),
        mean_raw_joint_f1=statistics.mean(r['metrics']['joint_time_bin_type_f1'] for r in rows),
        mean_raw_onset_f1=statistics.mean(r['metrics']['onset_f1'] for r in rows),
        mean_raw_grid_bonus=statistics.mean(r['metrics']['grid_bonus'] or 0 for r in rows),
        mean_tail_10s_recall=statistics.mean(r['metrics']['tail_10s_recall'] for r in rows),
        songs=rows, resources=budget.log_state())
    write_json(output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('songs','resources')}), flush=True)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def evaluate_with_fixed15(args, budget, device):
    original_evaluate(args, budget, device)
    # All four get the same e1000 baseline, not only the old release candidates.
    baseline = args.output.parent/'evaluation-e1000-fixed15'
    if not (baseline/'summary.json').is_file():
        evaluate15(args.continue_from, args.mode, args.variant, baseline, budget, device)
    evaluate15(args.checkpoint, args.mode, args.variant, args.output/'fixed15', budget, device)


if __name__ == '__main__':
    training.evaluate = evaluate_with_fixed15
    training.main()
