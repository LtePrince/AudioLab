"""Freeze the fixed 15-song comparison after every declared candidate finishes."""
from pathlib import Path
import hashlib, json, statistics, sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
from src.model_benchmark import rhythm_bonus
from src.product.artifacts import write_json

DEST = ROOT/'out/runtime_delivery_20261026'
plan = json.loads((DEST/'selection.json').read_text())
profile = json.loads((ROOT/'benchmarks/v1/protocol.json').read_text())['rhythm']
summaries = []; winners = {}; audio_winners = {}
for mode in ('4k', 'freex'):
    by_mode = []
    for candidate in plan['candidates'][mode]:
        folder = DEST/'evaluation'/candidate['candidate_id']
        summary = json.loads((folder/'summary.json').read_text())
        rows = []
        for i, song in enumerate(plan['songs']):
            source = folder/f'{i:02d}.json'
            value = json.loads(source.read_text());p=value['prediction'];r=value['reference']
            assert value['song']==song['id'] and value['checkpoint_sha256']==candidate['sha256']
            clock = dict(bpm=r['timebase_bpm'],beat_zero_seconds=r['offset_seconds'],
                         source='canonical reference chart clock',verified=False)
            grid = rhythm_bonus(p['onsets_seconds'],clock,profile)
            rows.append(dict(song=song['id'], status=p['status'], error=p.get('error'),
                onset_f1=value['metrics']['onset_f1'], joint_f1=value['metrics']['joint_time_bin_type_f1'],
                selection_joint_f1=value['metrics']['selection_joint_f1'],
                grid_bonus_per_onset=grid['bonus_per_onset'], note_count=p['note_count'], types=p['types'],
                reference_notes=value['metrics']['reference_notes'],
                prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        summary.update(songs=rows,
            mean_raw_joint_f1=statistics.mean(r['joint_f1'] for r in rows),
            mean_raw_onset_f1=statistics.mean(r['onset_f1'] for r in rows),
            mean_raw_grid_bonus=statistics.mean(r['grid_bonus_per_onset'] or 0 for r in rows))
        by_mode.append(summary);summaries.append(summary)
    winners[mode] = max(by_mode,key=lambda r:(r['mean_joint_f1'],r['completed'],r['mean_onset_f1']))['candidate']
    audio_winners[mode] = max((r for r in by_mode if r['candidate']['variant']=='audio'),
        key=lambda r:(r['mean_joint_f1'],r['completed'],r['mean_onset_f1']))['candidate']
result = dict(schema='audiolab-runtime-selection15-v1', scope=plan['scope'], selection=plan['selection'],
    seed=plan['seed'], songs=[dict(id=s['id'],audio_sha256=s['audio']['sha256'],duration=s['duration']) for s in plan['songs']],
    cache_sha256=plan['cache_sha256'], metric_sha256=plan['metric_sha256'],
    plan_sha256=hashlib.sha256((DEST/'selection.json').read_bytes()).hexdigest(),
    quality_status='experimental; generalization and playability unverified',
    metric_note='Selection/onset mean scores count unfinished generations as zero; raw metrics include partial failed output and are diagnostics only. Grid is a positive matching bonus, never decoding/quantization.',
    grid_profile=profile, candidates=summaries, winners=winners, audio_winners=audio_winners)
write_json(DEST/'selection_summary.json',result)
print(json.dumps(winners,indent=2))
