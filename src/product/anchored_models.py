"""Portable E14/E18 inference, with the training clock and group protocol intact."""
from __future__ import annotations

import math
import re

from src.anchored_ar import SCHEMA as MODEL_SCHEMA, FREEX_SCHEMA
from src.data.chart2array import LANE_CENTERS
from src.data.timing import tick_seconds

SCHEMA = 'audiolab-anchored-ar-runtime-v1'
EXPORT_POLICY = 'learned-stop-or-bounded-prefix-v1'
PARTIAL_TERMINATIONS = {
    'continued with no future audio interval': ('audio_end_cutoff',
        '模型未正常结束，已在音频末尾截断。已生成部分可预览和下载。'),
    'anchored decoder exhausted token budget before learned stop': ('token_limit_cutoff',
        '生成达到步骤上限，已保留前面生成的谱面，可预览和下载；后续音符可能缺失。'),
    'no valid group continuation': ('group_cutoff',
        '模型未能继续生成，已保留前面生成的谱面，可预览和下载；后续音符可能缺失。'),
}


def _partial(prediction):
    return (prediction['status'] == 'failed' and prediction['termination'] == 'failed'
            and prediction.get('error') in PARTIAL_TERMINATIONS)


def validate_contract(ck):
    c = ck['runtime_contract']
    mode = c.get('mode')
    if (c.get('schema') != SCHEMA or mode not in ('4k', 'freex')
            or ck.get('schema') != (MODEL_SCHEMA if mode == '4k' else FREEX_SCHEMA)
            or c.get('variant') not in ('audio', 'soft')
            or c['variant'] != ck.get('config', {}).get('variant')
            or c.get('layout_encoder') != 'full'
            or c.get('feature_storage_dtype') != 'float16'
            or c.get('onset_chunk_frames') != 1024
            or c.get('onset_context_frames') != 128
            or c.get('forced_skeleton') is not False
            or c.get('decoding') not in ('greedy', 'nucleus')
            or c.get('termination_policy') != 'learned_stop_required'
            or c.get('repair_profile') != 'none-model-output-v1'
            or not isinstance(c.get('model_label'), str) or not c['model_label']
            or not re.fullmatch('[0-9a-f]{64}', c.get('onset_checkpoint_sha256', ''))):
        raise ValueError('unsupported anchored AR runtime contract')
    return c


def phigros_notes(prediction, mode, bpm, offset, duration, *, allow_partial=False):
    """Change units only: no tick rounding, onset filtering or type/position repair."""
    if not ((prediction['status'] == 'completed' and prediction['termination'] == 'learned_stop')
            or (allow_partial and _partial(prediction))):
        raise ValueError('anchored generation did not finish: ' + prediction.get('error', 'missing learned stop'))
    if prediction.get('incomplete_coordinate_notes', 0) and not (allow_partial and _partial(prediction)):
        raise ValueError('anchored generation has incomplete coordinates')
    scale = tick_seconds(1, bpm)
    kinds = {'tap': 1, 'drag': 2, 'hold': 3, 'flick': 4}
    result = []
    for n in prediction['notes']:
        start, end, x = n['time_seconds'], n['end_seconds'], n['screen_x']
        if (not all(math.isfinite(v) for v in (start, end, x))
                or not offset <= start < duration or not start <= end <= duration + 1e-8):
            raise ValueError('anchored note lies outside the audio/offset timeline')
        kind = kinds[n['type']]
        if (kind == 3) != (end > start):
            raise ValueError('anchored Hold duration does not match note type')
        if mode == '4k':
            lane = n['position']
            if type(lane) is not int or not 0 <= lane < 4:
                raise ValueError('invalid anchored 4K lane')
            position = LANE_CENTERS[lane]
            if abs(.5 + position * .05625 - x) > 1e-7:
                raise ValueError('anchored lane and screen coordinate disagree')
        else:
            if not .05 <= x <= .95:
                raise ValueError('anchored coordinate outside visible interval')
            position = (x - .5) / .05625
        tick = (start-offset)/scale
        # The last representable audio second may round back to duration after
        # converting units. Keep this sub-ULP boundary inside the audio.
        while tick*scale+offset >= duration:
            tick = math.nextafter(tick, -math.inf)
        result.append(dict(type=kind, time=tick, holdTime=(end-start)/scale,
                           positionX=position, speed=1., floorPosition=tick*scale))
    return result


def export_prediction(prediction, mode, bpm, offset, duration):
    """Retain a valid prefix after a controlled stop, without changing research results."""
    details = dict(termination_policy=EXPORT_POLICY, generation_status=prediction['status'],
        generator_termination=prediction['termination'], generator_error=prediction.get('error'),
        termination=prediction['termination'], model_completed=prediction['status']=='completed',
        truncated=False, warnings=[])
    if not _partial(prediction):
        return phigros_notes(prediction, mode, bpm, offset, duration), details
    kept = []
    changes = dict(notes_after_audio=0, shortened_holds=0,
                   incomplete_notes=prediction.get('incomplete_coordinate_notes', 0))
    for original in prediction['notes']:
        n = dict(original)
        start, end = n['time_seconds'], n['end_seconds']
        if not all(math.isfinite(t) for t in (start, end)) or start < offset or end < start:
            raise ValueError('invalid anchored partial note timeline')
        if start >= duration:
            changes['notes_after_audio'] += 1
            continue
        if n['type'] == 'hold':
            if end == start:  # Interrupted before the first Hold-duration token.
                changes['incomplete_notes'] += 1
                continue
            if end > duration:
                n['end_seconds'] = duration
                changes['shortened_holds'] += 1
        kept.append(n)
    bounded = dict(prediction, notes=kept)
    notes = phigros_notes(bounded, mode, bpm, offset, duration, allow_partial=True)
    if not notes:
        raise ValueError('anchored generation did not finish and has no complete notes to preview')
    termination, message = PARTIAL_TERMINATIONS[prediction['error']]
    label = '定轨 4K' if mode == '4k' else '无轨 free-x'
    warning = dict(code=termination, mode=mode, message=f'{label}：{message}')
    details.update(termination=termination, truncated=True, model_completed=False,
                   audio_end_seconds=duration, cutoff_changes=changes, warnings=[warning])
    return notes, details
