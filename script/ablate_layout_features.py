"""One frozen song, two layouts, matched versus zero acoustic features.

No audio-candidate inference, reference teacher forcing, or product-default change.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from script.diagnose_patterns import compact, compact_comparison
from script.review_hold_repairs import digest, provenance, verified_source
from src.audio_understanding.inference import code_version, load_audio
from src.conditioning_audit import tensor_fingerprint
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.timing import tick_seconds
from src.pattern_diagnostics import chart_events, compare, onset_groups, summarize
from src.product.artifacts import chart_document, repair_log, validate_chart, write_json, write_preview
from src.product.pipeline import JobControl, decode_notes, quantize_candidates
from src.resource_budget import ResourceBudget, add_resource_args

SCHEMA = 'audiolab-sampled-layout-feature-ablation-v1'
CONDITIONS = ('matched', 'zero_acoustic_features')


def output_bytes(folder):
    return sum(p.lstat().st_size for p in folder.rglob('*') if p.is_file() or p.is_symlink())


def check_output(folder, budget):
    budget.check(force=True)
    size = output_bytes(folder)
    if size > 5*1024**2:
        raise RuntimeError('layout ablation exceeds 5 MiB artifact budget')
    return size


def compare_conditions(a, b):
    """Candidate groups match, sampled note IDs do not necessarily correspond."""
    ga, gb = onset_groups(a), onset_groups(b)
    if [g[0]['tick'] for g in ga] != [g[0]['tick'] for g in gb]:
        raise ValueError('condition outputs differ in fixed candidate coverage')
    changed = []
    for left, right in zip(ga, gb):
        def payload(group):
            return [dict(type=e['type'], screen_x=e['screen_x'],
                         duration_seconds=e['end_seconds']-e['time_seconds']) for e in group]
        x, y = payload(left), payload(right)
        if x != y:
            changed.append(dict(tick=left[0]['tick'], time_seconds=left[0]['time_seconds'], matched=x, zero=y))
    return dict(candidate_groups=len(ga), changed_candidate_groups=len(changed),
                matched_notes=len(a), zero_notes=len(b), examples=changed[:12],
                note_identity='independent sampled sequences; only candidate onset groups are paired')


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status='running', expected_rows=4, results=[], config=vars(args),
                  product_default_changed=False, human_playability='unreviewed', is_quality_holdout=False)
    control = None
    try:
        if args.threads not in (1, 2) or args.max_decoder_steps < 1:
            raise ValueError('one/two CPU threads and positive decoder budget required')
        torch.set_num_threads(args.threads)
        device = torch.device(args.device); budget = ResourceBudget(args, device, output)
        control = JobControl(output, budget, args.timeout_seconds)
        root = Path(__file__).resolve().parents[1]
        implementation = code_version()
        implementation['source_sha256'].update({str(p):digest(root/p) for p in map(Path, (
            'script/ablate_layout_features.py', 'script/train_models.py', 'src/product/pipeline.py',
            'src/conditioning_audit.py', 'src/pattern_diagnostics.py', 'src/models/chart_decoder.py',
            'src/models/attention.py', 'src/data/chart_constraints.py', 'src/data/chart_tokenizer.py'))})
        source_job = Path(args.source_job).resolve()
        manifest, source = verified_source(source_job, '4k')
        manifest_free, _ = verified_source(source_job, 'freex')
        if manifest != manifest_free or manifest['config']['layout_encoder'] != 'full':
            raise ValueError('both routes from the same full-encoder frozen job required')
        for mode in ('4k', 'freex'):
            route = manifest['routes'][mode]
            if (route['temperature'], route['top_p'], route['sampling_policy'], route['chord_size_cap']) != (1., .95, 'flat', None):
                raise ValueError('this experiment only reuses the existing default sampling configuration')
        config = manifest['config']; bpm, offset, seed = config['bpm'], config['offset'], config['seed']
        audio = Path(source['artifacts']['audio']['path'])
        evidence = json.loads(Path(source['artifacts']['candidates']['path']).read_text())
        skeleton, quantization = quantize_candidates(evidence, bpm, offset)
        control.update('audio_features')
        mel, duration = load_audio(audio, max_seconds=600, budget=budget)
        if abs(duration-manifest['audio']['duration_seconds']) > 1/22050:
            raise ValueError('decoded audio clock differs from the frozen source')
        report.update(implementation=implementation, source=source, source_config=config,
                      fixed_inputs=dict(audio=provenance(audio), candidates=source['artifacts']['candidates'],
                        mel=tensor_fingerprint(mel), duration_seconds=duration, bpm=bpm, offset=offset, seed=seed,
                        candidate_ticks=skeleton, quantization=quantization, temperature=1., top_p=.95,
                        sampling_policy='flat', max_chord_size=None, encoder='full-attention-v1',
                        termination='natural_eos_required'),
                      runtime=dict(torch=str(torch.__version__), device=str(device), threads=args.threads),
                      interpretation='sampled conditional sensitivity; zero retains time, biases and sampled history; no quality score')
        started = time.monotonic(); all_events = {}; previews = {condition:{} for condition in CONDITIONS}
        for mode, subdir in [('4k', 'decoder'), ('freex', 'decoder_freex')]:
            checkpoint = Path(args.model_root)/subdir/'decoder_best.pt'
            if digest(checkpoint) != manifest['model_sha256'][mode]:
                raise ValueError('checkpoint differs from frozen product')
            original_features = None
            for condition in CONDITIONS:
                row = dict(mode=mode, condition=condition, status='running')
                try:
                    check_output(output, budget); control.update(f'{mode}_{condition}')
                    folder = output/condition; folder.mkdir(exist_ok=True)
                    raw, decoding = decode_notes(mel, duration, skeleton, checkpoint, mode, bpm, offset, seed,
                                                 device, control, max_tokens=args.max_decoder_steps,
                                                 encoder_mode='full', feature_condition=condition)
                    intervention = decoding['feature_intervention']
                    fingerprint = (intervention['source_features'], intervention['valid_mask'])
                    if original_features is None: original_features = fingerprint
                    elif fingerprint != original_features:
                        raise ValueError('paired conditions did not use identical source encoder features/mask')
                    raw_chart = chart_document(raw, bpm, offset, folder/f'{mode}.raw.json')
                    end_tick = (duration-offset)/tick_seconds(1, bpm)
                    final, repairs = (constrain_4k_notes(raw, end_tick) if mode == '4k' else constrain_free_notes(raw, end_tick))
                    chart = chart_document(final, bpm, offset, folder/f'{mode}.json')
                    events, checks = validate_chart(chart, mode, duration, skeleton)
                    events['source_audio_sha256'] = source['artifacts']['audio']['sha256']
                    write_json(folder/f'{mode}.events.json', events)
                    repair_record = dict(profile='static-v1', summary=repairs, changes=repair_log(raw, final))
                    write_json(folder/f'{mode}.repairs.json', repair_record)
                    before, after = chart_events(raw_chart, mode), chart_events(chart, mode)
                    row.update(status='completed', checkpoint=provenance(checkpoint), decoding=decoding,
                               checks=checks, raw=compact(summarize(before)), final=compact(summarize(after)),
                               repair=compact_comparison(compare(before, after)), files={})
                    serialized = {'raw.json':raw_chart, 'json':chart, 'events.json':events, 'repairs.json':repair_record}
                    for suffix, expected in serialized.items():
                        path = folder/f'{mode}.{suffix}'
                        if json.loads(path.read_text()) != expected: raise ValueError('exported JSON readback differs')
                        row['files'][suffix] = dict(path=str(path.relative_to(output)), sha256=digest(path), bytes=path.stat().st_size)
                    if condition == 'matched':
                        frozen = manifest['routes'][mode]
                        row['frozen_baseline_bytes_equal'] = dict(
                            raw=digest(folder/f'{mode}.raw.json') == manifest['artifacts'][frozen['raw_file']]['sha256'],
                            final=digest(folder/f'{mode}.json') == manifest['artifacts'][frozen['chart_file']]['sha256'])
                    all_events[(mode, condition)] = dict(raw=before, final=after)
                    previews[condition][mode] = dict(events=events['events'], chart_file=f'{mode}.json', events_file=f'{mode}.events.json')
                except Exception as error:
                    row.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
                report['results'].append(row)
                gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
        report['comparisons'] = {mode:{phase:compare_conditions(all_events[(mode, CONDITIONS[0])][phase],
                                                                all_events[(mode, CONDITIONS[1])][phase])
                                      for phase in ('raw', 'final')}
                                 for mode in ('4k', 'freex') if all((mode,c) in all_events for c in CONDITIONS)}
        report['previews'] = []
        for condition, routes in previews.items():
            if len(routes) != 2: continue
            folder = output/condition; (folder/audio.name).symlink_to(audio)
            (folder/'manifest.json').symlink_to('../report.json')
            write_preview(folder/'preview.html', title=f'布局特征对照 · {condition} · 未经音乐性评价',
                          audio_name=audio.name, duration=duration, routes=routes)
            report['previews'].append(dict(condition=condition, path=f'{condition}/preview.html', audio_storage='unchanged source symlink'))
        completed = sum(r['status']=='completed' for r in report['results'])
        if any(digest(v['path']) != v['sha256'] for v in [source['manifest'], *source['artifacts'].values()]):
            raise ValueError('frozen source changed during generation')
        report.update(status='completed' if completed==4 else 'completed_with_failures', completed_rows=completed,
                      failed_rows=4-completed, elapsed_seconds=time.monotonic()-started, resources=budget.log_state())
        control.update(status='completed' if completed==4 else 'failed', stage='done')
        path = output/'report.json'
        for _ in range(4):
            write_json(path, report)
            if json.loads(path.read_text()) != report: raise ValueError('report readback differs')
            size = check_output(output, budget)
            if size == report.get('output_bytes'): break
            report['output_bytes'] = size
        else: raise ValueError('output size did not converge')
        return report
    except BaseException as error:
        (output/'report.json').unlink(missing_ok=True)
        for page in output.glob('*/preview.html'): page.unlink(missing_ok=True)
        if control is not None: control.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-job', 'model-root', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-decoder-steps', type=int, default=12000)
    p.add_argument('--timeout-seconds', type=float, default=900)
    add_resource_args(p); p.set_defaults(max_rss_gib=4)
    return p


if __name__ == '__main__':
    report = run(parser().parse_args())
    print(json.dumps({key:report[key] for key in ('status','completed_rows','failed_rows','elapsed_seconds','resources','output_bytes')}))
    raise SystemExit(0 if report['status']=='completed' else 1)
