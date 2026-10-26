"""Portable E14/E18 inference, with the training clock and group protocol intact."""
from __future__ import annotations

import math
import re

from src.anchored_ar import SCHEMA as MODEL_SCHEMA, FREEX_SCHEMA
from src.data.chart2array import LANE_CENTERS
from src.data.timing import tick_seconds

SCHEMA = 'audiolab-anchored-ar-runtime-v1'


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


def phigros_notes(prediction, mode, bpm, offset, duration):
    """Change units only: no tick rounding, onset filtering or type/position repair."""
    if prediction['status'] != 'completed' or prediction['termination'] != 'learned_stop':
        raise ValueError('anchored generation did not finish: ' + prediction.get('error', 'missing learned stop'))
    if prediction.get('incomplete_coordinate_notes', 0):
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
        result.append(dict(type=kind, time=(start-offset)/scale, holdTime=(end-start)/scale,
                           positionX=position, speed=1., floorPosition=(start-offset)))
    return result
