"""Whole-song teacher-forced diagnostics; no sampling or label repairs."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from src.audio_understanding import FRAME_SECONDS
from src.audio_understanding.inference import sha256
from src.data.chart2array import LANE_CENTERS, Phigros4kConvertor
from src.data.chart_tokenizer import ChartTokenizer, TokenNote, FREE_LO, FREE_HI, hit_to_bin
from src.data.timing import TIMING_SCHEMA, normalized_span, tick_seconds
from src.resource_budget import GIB, resource_snapshot

SCHEMA = 'audiolab-whole-song-conditioning-audit-v2'
CONDITIONS = ('matched', 'encoded_feature_reverse', 'wrong_song_resampled_features',
              'zero_acoustic_features')
BIN_WIDTH = (FREE_HI - FREE_LO) / 64


class ReferenceError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def tensor_fingerprint(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def file_fingerprint(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'non-finite {name}')
    return value


def _raw_records(document, duration):
    """Check native and quantized endpoints; never clamp the reference tail."""
    lines = document['judgeLineList']
    if not lines:
        raise ValueError('empty reference line list')
    bpm, offset = _finite(lines[0]['bpm'], 'bpm'), _finite(document.get('offset', 0), 'offset')
    tick_seconds(1, bpm)
    records, issues = [], []
    for li, line in enumerate(lines):
        source_bpm = _finite(line['bpm'], 'line bpm')
        tick_seconds(1, source_bpm)
        for side in ('notesAbove', 'notesBelow'):
            for ni, note in enumerate(line[side]):
                identity = f'line-{li}/{side}/note-{ni}'
                kind = _finite(note['type'], 'type')
                if kind not in (1, 2, 3, 4):
                    issues.append(dict(id=identity, reason='invalid_type')); continue
                kind = int(kind)
                native = _finite(note['time'], 'time')
                hold = _finite(note.get('holdTime', 0), 'holdTime') if kind == 3 else 0.
                pos = _finite(note['positionX'], 'positionX')
                tick, dur = normalized_span(native, hold, source_bpm, bpm)
                start = tick_seconds(native, source_bpm) + offset
                end = tick_seconds(native + hold, source_bpm) + offset
                qstart, qend = tick_seconds(tick, bpm)+offset, tick_seconds(tick+dur, bpm)+offset
                reason = ('negative_hold' if hold < 0 else
                          'negative_token_tick' if tick < 0 else
                          'native_outside_audio' if start < 0 or start >= duration or end > duration+1e-9 else
                          'quantized_outside_audio' if qstart < 0 or qstart >= duration or qend > duration+1e-9 else None)
                if reason:
                    issues.append(dict(id=identity, reason=reason, native_seconds=[start, end],
                                       quantized_seconds=[qstart, qend]))
                records.append(dict(id=identity, line_id=li, above=side=='notesAbove',
                                    type=kind, tick=tick, dur=dur, pos_x=pos,
                                    native_seconds=[start, end], quantized_seconds=[qstart, qend],
                                    source_bpm=source_bpm, native_tick=native, native_duration=hold))
    if issues:
        raise ReferenceError('complete reference is unsupported; no events were silently excluded', issues)
    if not records:
        raise ValueError('reference has no notes')
    return records, bpm, offset


def _link_key(record):
    return tuple(record[key] for key in ('line_id', 'above', 'tick', 'type', 'dur', 'pos_x'))


@dataclass
class Reference:
    tokens: torch.Tensor
    frames: torch.Tensor
    offsets: torch.Tensor
    offset_mask: torch.Tensor
    metadata: dict


def load_reference(chart_path, duration, *, gameplay_path=None, source_root=None):
    """Reuse the trained representation, trace every change, refuse incomplete inputs."""
    chart_path = Path(chart_path)
    chart = json.loads(chart_path.read_text())
    chart_source = file_fingerprint(chart_path)
    bpm, offset = float(chart['judgeLineList'][0]['bpm']), float(chart.get('offset', 0))
    provenance = dict(reference_chart=chart_source)
    if gameplay_path is None:
        raw, bpm, offset = _raw_records(chart, duration)
        rows = [dict(row) for row in raw]
        tk = ChartTokenizer(4)
        for row in rows:
            row['lane'] = Phigros4kConvertor._lane_from_x(row['pos_x'])
            row['offset'] = 0.
        source = 'derived-4k-chart-not-acoustic-truth'
    else:
        gameplay_path = Path(gameplay_path)
        meta_path = gameplay_path.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text())
        if meta.get('timing_schema') != TIMING_SCHEMA or not math.isclose(meta['bpm'], bpm) \
                or not math.isclose(meta['offset'], offset, abs_tol=1e-9) \
                or meta.get('jsonl_sha256') != sha256(gameplay_path):
            raise ValueError('mining timing/content metadata mismatch')
        original_path = Path(meta['source_path'])
        if not original_path.is_absolute():
            if source_root is None:
                raise ValueError('relative mining source_path requires explicit source_root')
            original_path = Path(source_root) / original_path
        original = file_fingerprint(original_path)  # missing source is never accepted
        if original['sha256'] != meta['source_sha256']:
            raise ValueError('mining source SHA mismatch')
        document = json.loads(Path(original['path']).read_text())
        raw, source_bpm, source_offset = _raw_records(document, duration)
        if not math.isclose(source_bpm, bpm) or not math.isclose(source_offset, offset, abs_tol=1e-9):
            raise ValueError('mined original clock differs from the reference clock')
        mined = [json.loads(line) for line in gameplay_path.read_text().splitlines() if line.strip()]
        if len(mined) != meta['notes'] or len(mined) != len(raw):
            raise ValueError('mined row count does not cover the complete original reference')
        provenance.update(gameplay=file_fingerprint(gameplay_path), mining_metadata=file_fingerprint(meta_path),
                          original_chart=original, relative_source_root=str(Path(source_root).resolve())
                          if source_root is not None else None)
        available = defaultdict(deque)
        for row in raw:
            available[_link_key(row)].append(row)
        ambiguous = sum(len(group) for group in available.values() if len(group) > 1)
        rows = []
        for index, note in enumerate(mined):
            for field in ('tick', 'dur', 'type', 'line_id'):
                value = _finite(note[field], field)
                if value != int(value):
                    raise ValueError(f'mined {field} must be integral')
            _finite(note['pos_x'], 'mined pos_x')
            x = _finite(note['hit_x'], 'mined hit_x')
            group = available[_link_key(note)]
            if not group:
                raise ValueError(f'mined row {index} has no matching source note identity fields')
            row = dict(group.popleft(), jsonl_row=index, hit_x=x)
            row['lane'], row['offset'] = hit_to_bin(x)
            rows.append(row)
        if any(available.values()):
            raise ValueError('original notes are missing from mined rows')
        provenance['source_link_policy'] = 'line/side/normalized-tick/type/duration/positionX multiset; FIFO ties'
        provenance['ambiguous_source_identity_rows'] = ambiguous
        source, tk = 'mined-free-x-with-source-hash-not-acoustic-truth', ChartTokenizer(64)

    counts = Counter(); changes = []; retained = {}; notes = []
    for row in rows:
        before, after = row['native_seconds'], row['quantized_seconds']
        actions = []
        if any(abs(a-b) > 1e-9 for a,b in zip(before, after)):
            counts['endpoint_quantized'] += 1; actions.append('endpoint_quantized')
        if tk.n_lanes == 4 and row['pos_x'] != float(LANE_CENTERS[row['lane']]):
            counts['position_quantized'] += 1; actions.append('position_quantized')
        if tk.n_lanes == 64 and (row['hit_x'] < FREE_LO or row['hit_x'] > FREE_HI-1e-9):
            counts['position_clamped'] += 1; actions.append('position_clamped')
        key = (row['tick'], row['lane'])
        if key in retained:
            counts['duplicate_tick_position_dropped'] += 1
            changes.append(dict(id=row['id'], actions=actions+['duplicate_tick_position_dropped'],
                                kept_source_id=retained[key]['id'], tick=key[0], position=key[1]))
            continue
        row = dict(row)
        if row['type'] == 3 and row['dur'] < 1:
            row['type'], row['dur'] = 1, 0
            counts['short_hold_to_tap'] += 1; actions.append('short_hold_to_tap')
        retained[key] = row
        notes.append(TokenNote(row['tick'], row['lane'], row['type'], row['dur']))
        if actions:
            changes.append(dict(id=row['id'], actions=actions, tick=key[0], position=key[1],
                                native_seconds=before, quantized_seconds=after))
    tokens = tk.encode_notes(notes)
    decoded = tk.decode_tokens(tokens, strict=True)
    if decoded != sorted(notes,key=lambda n:(n.tick,n.lane)):
        raise ValueError('tokenizer changed retained reference time/type/position/duration')
    ticks = tk.ticks_per_position(tokens)
    seconds = [tick_seconds(tick, bpm)+offset for tick in ticks]
    frames = torch.tensor([second/(FRAME_SECONDS*2) for second in seconds], dtype=torch.float32)
    omask = [tk.n_lanes == 64 and tk.kind(token)=='NOTE' for token in tokens]
    offsets = [retained[(tick,tk.value(token)//4)]['offset'] if masked else 0.
               for tick,token,masked in zip(ticks,tokens,omask)]
    tensors = dict(tokens=torch.tensor(tokens,dtype=torch.long), frames=frames,
                   offsets=torch.tensor(offsets,dtype=torch.float32), offset_mask=torch.tensor(omask))
    metadata = dict(label_source=source, provenance=provenance, complete_reference=True,
                    timebase_bpm=bpm, offset_seconds=offset, timing_source='reference-chart',
                    timing_schema=TIMING_SCHEMA, frame_seconds=FRAME_SECONDS*2,
                    source_notes=len(rows), retained_notes=len(notes), source_holds=sum(r['type']==3 for r in rows),
                    retained_types=dict(Counter(str(n.type) for n in notes)),
                    transformations=dict(counts), transformation_details=changes,
                    source_line_bpms=[float(line['bpm']) for line in
                                      (document if gameplay_path is not None else chart)['judgeLineList']],
                    native_reference_seconds=[min(r['native_seconds'][0] for r in rows),
                                              max(r['native_seconds'][1] for r in rows)],
                    quantized_reference_seconds=[min(r['quantized_seconds'][0] for r in rows),
                                                 max(r['quantized_seconds'][1] for r in rows)],
                    max_endpoint_quantization_error_seconds=max(abs(a-b) for r in rows
                        for a,b in zip(r['native_seconds'],r['quantized_seconds'])),
                    retained_note_roundtrip='exact tick/position/type/duration equality',
                    token_count=len(tokens), supervised_token_count=len(tokens)-1,
                    fingerprints={key:tensor_fingerprint(value) for key,value in tensors.items()})
    return Reference(**tensors, metadata=metadata)


def feature_geometry(mel_frames, duration, stride=2):
    if stride != 2 or mel_frames != math.ceil(duration/FRAME_SECONDS) or duration <= 0:
        raise ValueError('audio must cover the complete product feature clock with stride 2')
    padded = math.ceil(mel_frames/stride)*stride
    frames = padded//stride
    times = torch.arange(frames, dtype=torch.float64) * FRAME_SECONDS * stride
    valid = times < duration
    if not valid.any():
        raise ValueError('no valid audio memory')
    return valid[None], dict(raw_mel_frames=mel_frames, padded_mel_frames=padded,
                            memory_frames=frames, valid_frames=int(valid.sum()), duration_seconds=duration,
                            memory_frame_seconds=FRAME_SECONDS*stride,
                            memory_time_origin_seconds=0., last_memory_time_seconds=float(times[-1]),
                            padding='zero-to-smallest-stride-multiple', encoder='full-attention-v1')


def estimate_working_set(enc, dec, mel_frames, token_count):
    """Explicit math-attention fallback estimate, one inference layer at a time."""
    t = math.ceil(mel_frames/enc.mel_per_frame); length = token_count-1
    if t < 1 or length < 1:
        raise ValueError('empty input')
    eh = enc.conv_in.out_channels; dh = dec.tok_emb.embedding_dim
    heads = dec.blocks[0].h
    parts = dict(encoder_scores=2*enc.num_heads*t*t*4,
                 decoder_self_scores=2*heads*length*length*4,
                 decoder_cross_scores=2*heads*length*t*4,
                 cross_masks_and_distance=length*t*12,
                 logits_and_logprob=length*dec.lm_head.out_features*4*3,
                 features_hidden_mlp=(mel_frames*eh*20+(t+length)*dh*24)*4,
                 model_parameters=sum(p.numel()*p.element_size() for model in (enc,dec) for p in model.parameters()))
    return dict(method='float32-math-attention-two-score-matrices-v1', memory_frames=t,
                teacher_forcing_tokens=length, parts_bytes=parts, total_bytes=sum(parts.values()),
                inference_only=True, does_not_include_unbounded_runtime_overhead=True)


def enforce_preflight(estimate, budget, device):
    budget.check(force=True)
    snapshot = resource_snapshot(budget.directory)
    extra = estimate['total_bytes']
    # CPU retains full mel, reference and two feature snapshots even on GPU.
    cpu_extra = extra if device.type == 'cpu' else estimate['parts_bytes']['features_hidden_mlp']
    result = dict(**estimate, snapshot=snapshot, cpu_extra_bytes=cpu_extra,
                  limits=dict(max_rss_gib=budget.args.max_rss_gib,
                              min_available_ram_gib=budget.args.min_available_ram_gib,
                              gpu_memory_fraction=budget.args.gpu_memory_fraction))
    error = None
    if snapshot['rss_gib']+cpu_extra/GIB > budget.args.max_rss_gib or \
            snapshot['ram_available_gib']-cpu_extra/GIB < budget.args.min_available_ram_gib:
        error = 'full encoder/teacher-forcing estimate exceeds RAM budget'
    if device.type == 'cuda':
        free,total = torch.cuda.mem_get_info(device)
        available = min(free,total*budget.args.gpu_memory_fraction-torch.cuda.memory_allocated(device))
        result['gpu'] = dict(free_bytes=free,total_bytes=total,permitted_extra_bytes=available)
        if extra > available:
            error = 'full encoder/teacher-forcing estimate exceeds GPU budget'
    if error:
        raise ReferenceError(error, result)
    return result


@torch.inference_mode()
def encode_audio(enc, mel, duration, device):
    valid, geometry = feature_geometry(mel.shape[-1], duration, enc.mel_per_frame)
    padded = F.pad(mel, (0,geometry['padded_mel_frames']-mel.shape[-1]))
    features = enc.features(padded[None].to(device))
    if features.shape[:2] != valid.shape or not torch.isfinite(features).all():
        raise ValueError('non-finite or incomplete encoded features')
    return features, valid.to(device), dict(**geometry, features=tensor_fingerprint(features),
                                           valid_mask=tensor_fingerprint(valid))


def alter_features(features, valid, mode, *, other=None, other_valid=None):
    length = int(valid.sum())
    if valid.shape != features.shape[:2] or not length or not valid[0,:length].all() or valid[0,length:].any():
        raise ValueError('a nonempty contiguous valid prefix is required')
    result = features.clone()
    mapping = dict(condition=mode, target_valid_frames=length)
    if mode == 'encoded_feature_reverse':
        result[:,:length] = features[:,:length].flip(1)
        mapping.update(source_index='Ltarget-1-j', domain='encoded features, not waveform')
    elif mode == 'wrong_song_resampled_features':
        if other is None or other_valid is None:
            raise ValueError('wrong-song features required')
        other_length = int(other_valid.sum())
        if not other_length or not other_valid[0,:other_length].all() or other_valid[0,other_length:].any():
            raise ValueError('wrong song has no contiguous valid memory')
        result[:,:length] = F.interpolate(other[:,:other_length].transpose(1,2),size=length,
                                          mode='linear',align_corners=False).transpose(1,2)
        mapping.update(source_valid_frames=other_length,align_corners=False,
                       source_index='(j+0.5)*Lsource/Ltarget-0.5; edge values held',
                       temporal_mapping='duration-resampled encoded features')
    elif mode == 'zero_acoustic_features':
        result.zero_()
        mapping['remaining_conditioning'] = 'projection biases, memory time embedding, true token history'
    elif mode != 'matched':
        raise ValueError('unknown condition')
    return result,mapping


def token_scores(logits, targets, tk):
    if logits.shape != (len(targets), tk.vocab_size) or not torch.isfinite(logits).all():
        raise ValueError('invalid scoring logits')
    losses = -logits.float().log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)/math.log(2)
    first,end = tk.note(0,1),tk.note(0,1)+tk.n_lanes*4
    notes = (targets>=first)&(targets<end)
    masks = dict(all=targets!=0,time=(targets>=tk.dtb(1))&(targets<=tk.dtt(31)),note=notes,
                 duration=(targets>=tk.durb(1))&(targets<=tk.durt(31)))
    scores = {name:dict(count=int(mask.sum()),bits=float(losses[mask].mean()) if mask.any() else None)
              for name,mask in masks.items()}
    for kind in range(1,5):
        mask = notes & ((targets-first)%4==kind-1)
        scores[f'note_type_{kind}_token'] = dict(count=int(mask.sum()),bits=float(losses[mask].mean()) if mask.any() else None)
    for name in ('note_type','position'):
        scores[name] = dict(count=int(notes.sum()),bits=None)
    if notes.any():
        block = logits[notes,first:end].reshape(-1,tk.n_lanes,4)
        payload = targets[notes]-first
        for name,conditional,truth in [('note_type',block.logsumexp(1),payload%4),
                                       ('position',block.logsumexp(2),payload//4)]:
            scores[name]['bits'] = float(F.cross_entropy(conditional,truth)/math.log(2))
    return scores


def offset_scores(predicted, input_tokens, targets, mask, tk):
    note_mask = (input_tokens>=tk.note(0,1)) & (input_tokens<tk.note(0,1)+tk.n_lanes*4)
    if mask.shape != targets.shape or predicted.shape != targets.shape or not torch.equal(mask,note_mask):
        raise ValueError('offset mask must match consumed current NOTE positions')
    if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
        raise ValueError('non-finite offset')
    errors = (predicted-targets).abs()
    def score(selected):
        mae = float(errors[selected].mean()) if selected.any() else None
        return dict(count=int(selected.sum()),mae_sub_bin=mae,
                    mae_screen_width=mae*BIN_WIDTH if mae is not None else None)
    return dict(**score(mask),bin_width_screen_width=BIN_WIDTH,
                alignment='hidden after consuming current true NOTE; targets at same input index',
                conditioned_on='true reference bin and complete true token history',
                by_type={str(k):score(mask & ((input_tokens-tk.note(0,1))%4==k-1)) for k in range(1,5)})


@torch.inference_mode()
def score_condition(dec, acoustic, valid, reference):
    device = acoustic.device
    tokens,frames = reference.tokens.to(device),reference.frames.to(device)
    memory = dec.prepare_memory(acoustic)
    logits,hidden = dec(tokens[None,:-1],frames[None,:-1],memory,valid,return_hidden=True)
    tk = ChartTokenizer(dec.n_lanes)
    scores = token_scores(logits[0],tokens[1:],tk)
    offsets = None
    if dec.offset_head is not None:
        predicted = dec.predict_offset(hidden)[0]
        offsets = offset_scores(predicted,tokens[:-1],reference.offsets[:-1].to(device),
                                reference.offset_mask[:-1].to(device),tk)
    return dict(scores=scores,conditional_offset=offsets)


def _check_finite(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError('non-finite report value')
    if isinstance(value,dict):
        for key,item in value.items():
            if key == 'sha256' and (not isinstance(item,str) or len(item)!=64 or
                                    any(c not in '0123456789abcdef' for c in item)):
                raise ValueError('invalid report hash')
            _check_finite(item)
    if isinstance(value,list):
        for item in value: _check_finite(item)


def validate_report(report):
    if report.get('schema')!=SCHEMA or report.get('status') not in ('completed','completed_with_failures'):
        raise ValueError('invalid completed report schema/status')
    results=report['results'];expected=report['expected_rows']
    if len(results)!=expected or report['completed_rows']+report['failed_rows']!=expected:
        raise ValueError('incomplete result denominator')
    implementation=report['implementation']
    if not implementation['git_commit'] or type(implementation['working_tree_dirty']) is not bool \
            or not implementation['source_sha256']:
        raise ValueError('missing implementation provenance')
    if not all(key in report['data_list'] for key in ('path','sha256','bytes')) or not report['runtime']:
        raise ValueError('missing selection/runtime provenance')
    expected_pairs={(model,entry['index']) for model in ('decoder','decoder_freex') for entry in report['selection']}
    if len(expected_pairs)!=expected or {(row['model'],row['input']['index']) for row in results}!=expected_pairs:
        raise ValueError('missing or duplicated selected model/song row')
    completed=0
    for row in results:
        if row['status']=='failed':
            if not row.get('error'): raise ValueError('missing failure details')
            continue
        if row['status']!='completed' or set(row['conditions'])!=set(CONDITIONS):
            raise ValueError('missing conditions')
        completed+=1
        ref=row['reference'];counts=None
        def fingerprint(value,shape=None):
            if not isinstance(value,dict) or not all(key in value for key in ('sha256','shape','dtype')):
                raise ValueError('missing tensor fingerprint')
            if shape is not None and value['shape']!=shape:
                raise ValueError('fingerprint shape mismatch')
        for key in ('audio','wrong_audio'):
            if not all(k in row[key] for k in ('path','sha256','bytes')):
                raise ValueError('missing audio provenance')
        checkpoint=report['checkpoints'][row['model']]
        if not all(k in checkpoint for k in ('path','sha256','enc_config','dec_config','timing_schema')):
            raise ValueError('missing checkpoint provenance')
        required=['reference_chart']+(['gameplay','mining_metadata','original_chart']
                                      if row['model']=='decoder_freex' else [])
        for key in required:
            if not all(k in ref['provenance'][key] for k in ('path','sha256','bytes')):
                raise ValueError('missing reference provenance')
        if not ref['complete_reference'] or ref['token_count']!=ref['supervised_token_count']+1:
            raise ValueError('incomplete reference tokens')
        for key in ('tokens','frames','offsets','offset_mask'):
            fingerprint(ref['fingerprints'][key],[ref['token_count']])
        for key,mel_key in (('features','mel'),('wrong_features','wrong_mel')):
            feature=row[key];shape=feature['features']['shape']
            if len(shape)!=3 or shape[:2]!=[1,feature['memory_frames']] or feature['valid_frames']<1:
                raise ValueError('invalid feature dimensions')
            fingerprint(feature['features'],shape)
            fingerprint(feature['valid_mask'],[1,feature['memory_frames']])
            fingerprint(row[mel_key],[128,feature['raw_mel_frames']])
            valid,geometry=feature_geometry(feature['raw_mel_frames'],feature['duration_seconds'])
            if any(feature[k]!=v for k,v in geometry.items()) or tensor_fingerprint(valid)!=feature['valid_mask']:
                raise ValueError('feature timebase or mask changed')
        for value in row['conditions'].values():
            scores=value['scores'];current={k:v['count'] for k,v in scores.items()}
            fingerprint(value['features'],row['features']['features']['shape'])
            invariants={**{key:v['sha256'] for key,v in ref['fingerprints'].items()},
                        'valid_mask':row['features']['valid_mask']['sha256']}
            if value['fixed_inputs']!=invariants:
                raise ValueError('condition token/clock/offset/mask fingerprint changed')
            if scores['all']['count']!=ref['supervised_token_count'] or scores['note']['count']!=ref['retained_notes']:
                raise ValueError('reference score denominator mismatch')
            if scores['all']['count']!=sum(scores[k]['count'] for k in ('time','note','duration'))+1:
                raise ValueError('time/note/duration/EOS denominator mismatch')
            if any(scores[k]['count']!=ref['retained_notes'] for k in ('note_type','position')):
                raise ValueError('conditional NOTE denominator mismatch')
            if sum(scores[f'note_type_{k}_token']['count'] for k in range(1,5))!=ref['retained_notes']:
                raise ValueError('per-type NOTE denominator mismatch')
            if counts is not None and current!=counts: raise ValueError('condition denominators changed')
            counts=current
            if row['model']=='decoder_freex' and value['conditional_offset']['count']!=ref['retained_notes']:
                raise ValueError('conditional offset denominator mismatch')
            for score in scores.values():
                if type(score['count']) is not int or score['count']<0 or (score['bits'] is None)!=(score['count']==0):
                    raise ValueError('invalid score count or missing score')
            if row['model']=='decoder_freex':
                offsets=value['conditional_offset']
                if offsets['bin_width_screen_width']!=BIN_WIDTH or \
                        sum(v['count'] for v in offsets['by_type'].values())!=ref['retained_notes'] or \
                        offsets['mae_screen_width']!=offsets['mae_sub_bin']*BIN_WIDTH:
                    raise ValueError('invalid conditional offset units or per-type count')
        if row['conditions']['matched']['features']!=row['features']['features']:
            raise ValueError('matched feature fingerprint changed')
        if row['audio']['sha256']==row['wrong_audio']['sha256']:
            raise ValueError('wrong-audio control reuses identical audio')
    if completed!=report['completed_rows'] or (report['status']=='completed')!=(report['failed_rows']==0):
        raise ValueError('terminal status contradicts failures')
    _check_finite(report)


def atomic_report(path, report):
    """Read the actual final file back; any tail failure removes completion."""
    path=Path(path);temporary=path.with_suffix('.tmp')
    validate_report(report)
    try:
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        os.replace(temporary,path)
        restored=json.loads(path.read_text())
        validate_report(restored)
        if restored!=report:
            raise ValueError('report changed during serialization')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
