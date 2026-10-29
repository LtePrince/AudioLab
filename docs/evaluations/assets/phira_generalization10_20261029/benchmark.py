"""Frozen, inference-only Phira community evaluation; no training or grid repair.

Reference RPE timing/types follow TeamFlos/phira prpr/src/parse/rpe.rs and
prpr/src/scene/game.rs: audio time = beat*60/BPM + META.offset/1000 + info.offset.
Only constant-tempo charts are admitted. Moving-line coordinates are deliberately
not treated as a static 4K/free-x target. Chart correspondence is diagnostic.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import sys
import unicodedata
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
sys.path.insert(0, str(ROOT))
ASSETS = Path(__file__).resolve().parent
OUT = ROOT/'out/phira_generalization10_20261029'
IDS = [40717, 37396, 6990, 30114, 23968, 51352, 64888, 51818, 52417, 52244]
KINDS = {1: 'tap', 2: 'hold', 3: 'flick', 4: 'drag'}
CANDIDATES = {
    'E14-A': ('4k', 'audio', 1400, 'out/ar_saturation_20261023/continued1400/audio/model_e1400.pt'),
    'E14-S': ('4k', 'soft', 1400, 'out/ar_saturation_20261023/continued1400/soft/model_e1400.pt'),
    'E18-A': ('freex', 'audio', 1300, 'out/freex_anchored_20261025/continued1400/audio/model_e1300.pt'),
    'E18-S': ('freex', 'soft', 1400, 'out/freex_anchored_20261025/continued1400/soft/model_e1400.pt'),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from src.product.artifacts import write_json
    write_json(path, value)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalized_name(name):
    text = unicodedata.normalize('NFKC', name).casefold()
    return ''.join(c for c in text if c.isalnum())


def beat(value):
    if len(value) != 3 or value[2] == 0:
        raise ValueError('invalid RPE beat fraction')
    return value[0] + value[1]/value[2]


def reference_rpe(chart, metadata):
    tempos = chart['BPMList']
    if not tempos or abs(beat(tempos[0]['startTime'])) > 1e-9:
        raise ValueError('reference BPM must begin at beat zero')
    bpm = float(tempos[0]['bpm'])
    if bpm <= 0 or any(abs(float(t['bpm'])-bpm) > 1e-6 for t in tempos):
        raise ValueError('variable BPM is outside this first cohort')
    offset = float(chart['META']['offset'])/1000 + float(metadata.get('offset') or 0)
    notes, fake, hidden, unusual_size = [], 0, 0, 0
    for line_index, line in enumerate(chart['judgeLineList']):
        for n in line.get('notes') or []:
            if n.get('isFake', 0):
                fake += 1
                continue
            kind = KINDS[n['type']]
            start = beat(n['startTime'])*60/bpm + offset
            end = beat(n['endTime'])*60/bpm + offset if kind == 'hold' else start
            if not all(math.isfinite(v) for v in (start, end)) or end < start:
                raise ValueError('invalid reference time')
            hidden += n.get('alpha', 255) < 200
            unusual_size += abs(n.get('size', 1)-1) > .5
            notes.append(dict(time_seconds=start, end_seconds=end, type=kind,
                              source_line=line_index, source_position_x=n['positionX']))
    notes.sort(key=lambda n: (n['time_seconds'], n['source_line'], n['source_position_x']))
    if not notes or hidden or unusual_size:
        raise ValueError('reference has no real notes or has hidden/oversized judgment notes')
    return notes, dict(bpm=bpm, beat_zero_seconds=offset, source='Phira RPE BPMList + META.offset + info.offset',
                      verified=False, conversion_verified_against_source=True), dict(
        real_notes=len(notes), decorative_fake_notes_excluded=fake, low_alpha_real_notes=hidden,
        unusual_size_real_notes=unusual_size, lines=len(chart['judgeLineList']),
        type_counts=dict(Counter(n['type'] for n in notes)),
        max_simultaneous_notes=max(Counter(round(n['time_seconds'], 6) for n in notes).values()),
        touch_playtest='not_executed', coordinates='not scored: original moving-line geometry is not a static chart target')


def prepare():
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from src.data.audio2mel import AudioGPUprocessor
    os.chdir(ROOT)
    if (ASSETS/'selection.json').exists():
        raise FileExistsError('Frozen selection already exists; do not reselect after inference')
    discovery = read(OUT/'discovery.json')
    entries = {r['id']: r for r in discovery['candidates']}
    training_path = ROOT/'out/full_corpus_v1/cache/cache.json'
    training = read(training_path)
    names = {normalized_name(r['id'].removesuffix('_IN')) for r in training['rows']}
    hashes = {r['audio']['sha256'] for r in training['rows']}
    songs = []
    processor = AudioGPUprocessor(device='cpu')
    torch.set_num_threads(2)
    for id in IDS:
        entry = entries[id]
        assert entry['reviewed'] and 'regular' in entry['tags'] and 13 <= entry['difficulty'] < 16
        assert normalized_name(entry['name']) not in names, 'training title overlap'
        package = OUT/f'{id}.zip'
        assert shutil.disk_usage(OUT).free/2**30 > 8.25
        dest = OUT/str(id)
        dest.mkdir(exist_ok=True)
        package_hash = sha(package)
        with zipfile.ZipFile(package) as archive:
            info_entry = archive.getinfo('info.yml')
            assert info_entry.file_size < 65536
            metadata_bytes = archive.read(info_entry)
            metadata = yaml.safe_load(metadata_bytes)
            for name, limit in [(metadata['chart'], 12*2**20), (metadata['music'], 32*2**20)]:
                member = archive.getinfo(name)
                assert 0 < member.file_size < limit and not member.flag_bits & 1
            chart_bytes = archive.read(metadata['chart'])
            chart = json.loads(chart_bytes.decode('utf-8-sig'))
            notes, clock, screening = reference_rpe(chart, metadata)
            source_audio = dest/('source'+Path(metadata['music']).suffix.lower())
            with archive.open(metadata['music']) as source, source_audio.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
        audio_hash = sha(source_audio)
        assert audio_hash not in hashes, 'training audio byte overlap'
        wave = processor.load_from_path(source_audio).mean(dim=1, keepdim=True)
        assert torch.isfinite(wave).all()
        duration = wave.shape[-1]/22050
        assert 45 <= duration <= 300 and min(n['time_seconds'] for n in notes) >= 0
        assert max(n['end_seconds'] for n in notes) <= duration+.002
        audio = dest/'audio.wav'
        # Preserve codec/resampling overshoots; integer PCM would clip samples.
        # This exact mono float waveform is shared by inference and preview.
        sf.write(audio, wave.flatten().numpy(), 22050, subtype='FLOAT')
        decoded, sr = sf.read(audio, dtype='float32')
        assert sr == 22050 and len(decoded) == wave.shape[-1]
        assert np.array_equal(decoded, wave.flatten().numpy())
        (dest/'reference.rpe.json').write_bytes(chart_bytes)
        (dest/'info.yml').write_bytes(metadata_bytes)
        write(dest/'reference.json', dict(notes=notes, clock=clock, screening=screening))
        write(dest/'community.json', entry)
        songs.append(dict(id=id, name=entry['name'], difficulty=round(entry['difficulty'], 1),
            level=entry['level'], composer=entry['composer'], charter=entry['charter'],
            community_url=f'https://api.phira.cn/chart/{id}', download_url=entry['file'],
            reviewed=entry['reviewed'], rating=entry['rating'], rating_count=entry.get('ratingCount'),
            original_package_sha256=package_hash, source_audio_sha256=audio_hash,
            source_audio=str(source_audio.relative_to(ROOT)), chart_sha256=sha(dest/'reference.rpe.json'),
            audio=str(audio.relative_to(ROOT)), audio_sha256=sha(audio), duration_seconds=duration,
            reference=str((dest/'reference.json').relative_to(ROOT)), reference_sha256=sha(dest/'reference.json'),
            clock=clock, screening=screening, training_title_overlap=False, training_audio_byte_overlap=False))
        print(json.dumps(dict(prepared=id, name=entry['name'], duration=round(duration, 2),
                              bpm=clock['bpm'], notes=len(notes)), ensure_ascii=False), flush=True)
        del wave, decoded, chart
    candidates = []
    for id, (mode, variant, epoch, path) in CANDIDATES.items():
        digest = sha(ROOT/path)
        assert digest == read((ROOT/path).with_suffix('.json'))['sha256']
        candidates.append(dict(id=id, mode=mode, variant=variant, epoch=epoch, checkpoint=path, sha256=digest))
    sources = ['src/anchored_ar.py', 'src/model_benchmark.py', 'src/audio_understanding/evaluation.py',
               'script/train_grounded_ar.py', 'src/data/audio2mel.py', 'src/product/anchored_models.py']
    selection = dict(schema='audiolab-phira-generalization10-v1', experiment='E20',
        selected_at_utc=datetime.now(timezone.utc).isoformat(), status='frozen_before_inference',
        selection='13.x–15.x, reviewed regular charts; constant BPM; no low-alpha or oversized judgment notes; selection independent of model output',
        scope='new titles and original audio bytes relative to 300-song decoder training; no robust acoustic-fingerprint search; frozen frontend pretraining overlap not established',
        training_manifest_sha256=sha(training_path), training_updates=0,
        controls=dict(greedy=True, seed=42, max_tokens=12000, bpm='reference chart', offset='reference scoring only; not passed to generator',
                      forced_grid=False, repair=False, checkpoint_selection='fixed from prior training-song results'),
        metric_note='Same unique-onset matcher at25/50/100ms and same rhythm bonus. New time+type F1 ignores moving-line coordinates and is NOT the old time+bin+type joint F1. Reference is one valid authored chart, not the only valid arrangement.',
        source_sha256={p:sha(ROOT/p) for p in sources}, frozen_frontends=training['models'],
        rhythm_protocol=read(ROOT/'benchmarks/v1/protocol.json')['rhythm'], candidates=candidates, songs=songs)
    write(ASSETS/'selection.json', selection)
    # Keep only useful original chart/audio/metadata, not unused storyboards/videos.
    removed = []
    for p in sorted(OUT.glob('*.zip')):
        removed.append(dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)))
        p.unlink()
    write(OUT/'download_cleanup.json', dict(scope='only archives downloaded for this experiment', removed=removed))


def score(reference, prediction, clock, profile):
    from src.audio_understanding.evaluation import match_events
    from src.model_benchmark import rhythm_bonus, pattern_metrics
    rg, pg = defaultdict(list), defaultdict(list)
    for n in reference: rg[round(n['time_seconds'], 6)].append(n)
    for n in prediction['notes']: pg[round(n['time_seconds'], 6)].append(n)
    rt, pt = sorted(rg), sorted(pg)
    output = {}
    for tolerance in (25, 50, 100):
        pairs = match_events(rt, pt, tolerance/1000)
        output[f'onset_f1_{tolerance}ms'] = 2*len(pairs)/(len(rt)+len(pt)) if rt or pt else 1.
    pairs = match_events(rt, pt, .05)
    matched_types = sum(sum((Counter(n['type'] for n in rg[r]) & Counter(n['type'] for n in pg[p])).values()) for r,p in pairs)
    raw = 2*matched_types/(len(reference)+len(prediction['notes']))
    last = max(rt)
    tail = [t for t in rt if t >= last-10]
    errors = [1000*abs(a-b) for a,b in pairs]
    pred_grid = rhythm_bonus(pt, clock, profile)
    ref_grid = rhythm_bonus(rt, clock, profile)
    output.update(time_type_f1=raw, completed_only_time_type_f1=raw if prediction['status']=='completed' else 0.,
        grid_bonus=pred_grid['bonus_per_onset'] or 0., reference_grid_bonus=ref_grid['bonus_per_onset'],
        reference_notes=len(reference), predicted_notes=len(prediction['notes']),
        reference_unique_onsets=len(rt), predicted_unique_onsets=len(pt),
        note_count_ratio=len(prediction['notes'])/len(reference),
        matched_onset_mae_ms=statistics.mean(errors) if errors else None,
        last_onset_delta_seconds=pt[-1]-last if pt else None,
        tail_10s_recall=sum(r >= last-10 for r,p in pairs)/len(tail),
        reference_types=dict(Counter(n['type'] for n in reference)),
        predicted_types=dict(Counter(n['type'] for n in prediction['notes'])),
        generated_patterns=pattern_metrics(prediction['notes']))
    return output


def evaluate(mode, device_name):
    import torch
    from types import SimpleNamespace
    from script.train_grounded_ar import load_inputs, encode_waveform
    from src.data.audio2mel import AudioGPUprocessor
    from src.anchored_ar import AnchoredAR, generate
    from src.resource_budget import ResourceBudget
    from src.product.anchored_models import export_prediction
    from src.product.artifacts import chart_document, validate_chart, write_preview
    from src.product.phira import build_phira_package
    os.chdir(ROOT)
    torch.set_num_threads(2)
    plan = read(ASSETS/'selection.json')
    for path, digest in plan['source_sha256'].items(): assert sha(path) == digest, path
    for item in plan['frozen_frontends'].values(): assert sha(item['path']) == item['sha256']
    device = torch.device(device_name)
    args = SimpleNamespace(min_free_disk_gib=8, min_available_ram_gib=8, max_rss_gib=4,
        gpu_memory_fraction=.65, keep_checkpoints=1, model_root=ROOT/'checkpoints/timing_v1', mode=mode)
    budget = ResourceBudget(args, device, OUT)
    encoder, onset = load_inputs(args, device)
    models = {}
    for candidate in plan['candidates']:
        if candidate['mode'] != mode: continue
        assert sha(candidate['checkpoint']) == candidate['sha256']
        ck = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=True)
        assert ck['epoch']==candidate['epoch'] and ck['config']['variant']==candidate['variant']
        model = AnchoredAR(ck['config']['decoder_config']).to(device).eval()
        model.load_state_dict(ck['model'], strict=True)
        models[candidate['id']] = (candidate, model)
        del ck
    processor = AudioGPUprocessor(device='cpu')
    records = []
    with torch.inference_mode():
        for song in plan['songs']:
            budget.check(force=True)
            assert sha(song['audio'])==song['audio_sha256'] and sha(song['reference'])==song['reference_sha256']
            audio = ROOT/song['audio']
            waveform = processor.load_from_path(audio)
            view = encode_waveform(waveform, encoder, onset, budget, device)
            assert abs(view['duration']-song['duration_seconds']) < 1e-8
            reference = read(song['reference'])
            for run_id, (candidate, model) in models.items():
                dest = OUT/str(song['id'])/run_id
                dest.mkdir(exist_ok=True)
                report_path = dest/'result.json'
                if report_path.exists():
                    saved=read(report_path)
                    assert saved['selection_sha256']==sha(ASSETS/'selection.json')
                    records.append(saved)
                    continue
                prediction=generate(model, view, song['clock']['bpm'], use_soft=candidate['variant']=='soft',
                    greedy=True, seed=42, max_tokens=12000, check=budget.check)
                prediction.pop('trace', None)
                metrics=score(reference['notes'], prediction, song['clock'], plan['rhythm_protocol'])
                artifact_error=None
                try:
                    notes, details=export_prediction(prediction, mode, song['clock']['bpm'], 0., view['duration'])
                    stem=f'{song["id"]}_{mode}_{run_id}_20261029'
                    chart_path=dest/f'{stem}.json'
                    chart=chart_document(notes, song['clock']['bpm'], 0., chart_path)
                    events, checks=validate_chart(chart, mode, view['duration'], None)
                    events_path=dest/f'{stem}.events.json';write(events_path, events)
                    package_path=dest/f'{stem}.phira.zip'
                    package=build_phira_package(chart_path,audio,package_path,mode=mode,
                        title=f'{song["name"]} / {run_id} e{candidate["epoch"]}',duration_seconds=view['duration'],
                        source=dict(experiment='E20',checkpoint_sha256=candidate['sha256'],
                            community_chart_id=song['id'],source_audio_sha256=song['source_audio_sha256'],
                            model_completed=prediction['status']=='completed'),check=budget.check)
                    write_preview(dest/'preview.html',title=f'{song["name"]} / {run_id} e{candidate["epoch"]}',
                        audio_name='../audio.wav',duration=view['duration'],
                        tempo=dict(timebase_bpm=song['clock']['bpm'],source='provided'),
                        routes={mode:dict(events=events['events'],chart_file=chart_path.name,events_file=events_path.name,
                            package_file=package_path.name,model_label=f'{run_id} e{candidate["epoch"]}',warnings=details['warnings'])})
                    write(dest/'manifest.json',dict(generation=details,checks=checks,package=package))
                except ValueError as exc:
                    artifact_error=str(exc)
                result=dict(song_id=song['id'],name=song['name'],run_id=run_id,mode=mode,
                    epoch=candidate['epoch'],checkpoint_sha256=candidate['sha256'],
                    selection_sha256=sha(ASSETS/'selection.json'),status=prediction['status'],
                    error=prediction.get('error'),metrics=metrics,prediction=prediction,
                    artifact_error=artifact_error,resources=budget.log_state())
                write(report_path,result);records.append(result)
                write(OUT/f'progress-{mode}.json',dict(done=len(records),total=20,last=song['id'],
                    run=run_id,status=prediction['status'],resources=budget.log_state()))
                print(json.dumps(dict(done=len(records),total=20,run=run_id,song=song['name'],
                    status=prediction['status'],onset_f1=round(metrics['onset_f1_50ms'],4),
                    grid=round(metrics['grid_bonus'],4),notes=metrics['predicted_notes'],
                    artifact_error=artifact_error),ensure_ascii=False),flush=True)
            del waveform,view
            gc.collect()
    write(OUT/f'completed-{mode}.json',dict(status='completed',outputs=len(records),resources=budget.log_state()))


def summarize():
    plan=read(ASSETS/'selection.json')
    rows=[]
    for song in plan['songs']:
        for candidate in plan['candidates']:
            p=OUT/str(song['id'])/candidate['id']/'result.json'
            r=read(p)
            assert r['selection_sha256']==sha(ASSETS/'selection.json')
            rows.append({k:v for k,v in r.items() if k!='prediction'})
    aggregates=[]
    for candidate in plan['candidates']:
        subset=[r for r in rows if r['run_id']==candidate['id']]
        keys=['onset_f1_25ms','onset_f1_50ms','onset_f1_100ms','time_type_f1',
            'completed_only_time_type_f1','grid_bonus','reference_grid_bonus','note_count_ratio','tail_10s_recall']
        aggregates.append(dict(**candidate,total=len(subset),completed=sum(r['status']=='completed' for r in subset),
            **{f'mean_{k}':statistics.mean(r['metrics'][k] for r in subset) for k in keys},
            failed_songs=[r['name'] for r in subset if r['status']!='completed'],
            artifact_errors=[r['name'] for r in subset if r['artifact_error']]))
    summary=dict(schema='audiolab-phira-generalization10-results-v1',experiment='E20',
        observed_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(ASSETS/'selection.json'),
        metrics_scope=plan['metric_note'],training_updates=0,aggregates=aggregates,rows=rows)
    write(ASSETS/'summary.json',summary)
    header='<meta charset="utf-8"><title>AudioLab Phira 新歌测试</title><style>body{font:16px system-ui;margin:32px;background:#111725;color:#eee}a{color:#8cd8ff}td,th{padding:12px;border-bottom:1px solid #445}table{border-collapse:collapse}</style>'
    body='<h1>Phira 社区 10 首新歌测试</h1><p>参考谱难度 13–15；生成谱未定级。A 为纯音频，S 增加软起音。点击预览可听音频、下载本地 Phira 包；未正常结束时保留有效前缀并提示。</p><table><tr><th>曲目</th><th>参考难度</th>'+''.join(f'<th>{html.escape(c["id"])} e{c["epoch"]}</th>' for c in plan['candidates'])+'</tr>'
    for song in plan['songs']:
        body+=f'<tr><td>{html.escape(song["name"])}</td><td>{song["difficulty"]}</td>'
        for candidate in plan['candidates']:
            row=next(r for r in rows if r['song_id']==song['id'] and r['run_id']==candidate['id'])
            relative=f'{song["id"]}/{candidate["id"]}/preview.html'
            label='正常结束' if row['status']=='completed' else '截断前缀'
            body+=f'<td><a href="{relative}">预览</a><br>{label}<br>起点 F1 {row["metrics"]["onset_f1_50ms"]:.3f} / 网格 {row["metrics"]["grid_bonus"]:.3f}</td>' if not row['artifact_error'] else f'<td>无可预览音符；见 result.json</td>'
        body+='</tr>'
    (OUT/'index.html').write_text(header+body+'</table>')
    print(json.dumps(aggregates,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate','summarize'])
    parser.add_argument('--mode',choices=['4k','freex'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='evaluate':evaluate(args.mode,args.device)
    else:summarize()
