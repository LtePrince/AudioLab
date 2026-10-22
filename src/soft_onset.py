"""Opt-in dense audio-onset conditioning experiment; no target-derived inputs."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.audio_understanding import FRAME_SECONDS
from src.conditioning_audit import offset_scores, token_scores
from src.data.chart_tokenizer import ChartTokenizer, EOS, PAD
from src.data.timing import TIMING_SCHEMA, check_checkpoint_timing
from src.models.chart_decoder import ChartDecoder
from src.models.chart_history import (HISTORY_DIM, HISTORY_SCHEMA, RELATION_NAMES, RELATION_SCHEMA,
                                      history_features, relation_targets)
from src.resource_budget import GIB, resource_snapshot

SCHEMA = 'audiolab-soft-onset-memory-v1'
CACHE_SCHEMA = 'audiolab-frozen-soft-onset-cache-v1'
MEMORY_SECONDS = FRAME_SECONDS * 2


def dense_condition(arrays, memory_frames):
    """Keep every native-rate prediction, including low scores and tail offsets.

    Current frozen lane head has the same rate as decoder memory. A future
    higher-rate head must get an explicitly versioned representation; averaging
    or dropping its subframes here would silently lose onset evidence.
    """
    times = np.asarray(arrays['frame_times_seconds'], dtype=np.float64)
    expected = np.arange(memory_frames, dtype=np.float64) * MEMORY_SECONDS
    if times.shape != expected.shape or not np.allclose(times, expected, rtol=0, atol=1e-9):
        raise ValueError('soft onset head must match every encoder-memory frame; no resampling allowed')
    p = np.asarray(arrays['event_probability'], dtype=np.float64)
    delta = np.asarray(arrays['time_offset_seconds'], dtype=np.float64) / MEMORY_SECONDS
    if p.shape != times.shape or delta.shape != times.shape or \
            not np.isfinite(p).all() or not np.isfinite(delta).all() or \
            np.any((p < 0) | (p > 1)) or np.any((delta < -1e-8) | (delta > 1+1e-8)):
        raise ValueError('invalid dense onset probability/subframe offset')
    if not np.allclose(np.asarray(arrays['event_times_seconds']), times+delta*MEMORY_SECONDS,
                       rtol=0, atol=1e-9):
        raise ValueError('onset offset and absolute event clock differ')
    return torch.tensor(np.stack([p, delta], -1), dtype=torch.float32)[None]


def condition_input(values, condition):
    if condition not in ('none', 'soft', 'soft_history', 'soft_history_aux'):
        raise ValueError('unknown soft-onset experiment condition')
    return torch.zeros_like(values) if condition == 'none' else values


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_decoder(checkpoint, seed=42, *, with_history=False, with_relation_aux=False):
    """Explicit transfer: legacy state strict, then one declared zero residual."""
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint_timing(ck)
    if ck.get('soft_onset_schema') or ck['dec_config'].get('soft_onset_dim', 0):
        raise ValueError('initialization requires the frozen legacy decoder checkpoint')
    torch.manual_seed(seed)
    config = dict(ck['dec_config'], soft_onset_dim=2)
    if with_history:
        config['history_dim']=HISTORY_DIM
    if with_relation_aux:
        config['relation_auxiliary']=True
    decoder = ChartDecoder(**config)
    old = {k.removeprefix('decoder.'): v for k,v in ck['model'].items() if k.startswith('decoder.')}
    state = dict(old, **{'soft_onset_proj.weight': torch.zeros_like(decoder.soft_onset_proj.weight)})
    if with_history:
        state['history_proj.weight']=torch.zeros_like(decoder.history_proj.weight)
    if with_relation_aux:
        state.update({'relation_head.weight':torch.zeros_like(decoder.relation_head.weight),
                      'relation_head.bias':torch.zeros_like(decoder.relation_head.bias)})
    decoder.load_state_dict(state, strict=True)
    return decoder, config, dict(old_decoder_state_sha256=state_hash(old),
                                 initialized_state_sha256=state_hash(decoder.state_dict()),
                                 added_parameters=['soft_onset_proj.weight']+(['history_proj.weight'] if with_history else [])+
                                    (['relation_head.weight','relation_head.bias'] if with_relation_aux else []),
                                 initialization='soft/history projections: zero weights, no bias; '
                                     'optional relation head: zero weights and zero bias')


def load_decoder(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return decoder_from_checkpoint(ck, device), ck


def decoder_from_checkpoint(ck, device):
    """Strict shared loader for research and explicitly packaged runtime models."""
    check_checkpoint_timing(ck)
    if ck.get('soft_onset_schema') != SCHEMA or ck.get('condition') not in ('none','soft','soft_history','soft_history_aux') \
            or ck.get('dec_config',{}).get('soft_onset_dim') != 2:
        raise ValueError('not a supported explicit soft-onset experiment checkpoint')
    has_history=ck['dec_config'].get('history_dim',0)==HISTORY_DIM
    if has_history != ck['condition'].startswith('soft_history') or (has_history and ck.get('history_schema')!=HISTORY_SCHEMA):
        raise ValueError('checkpoint history schema/condition mismatch')
    has_aux=ck['dec_config'].get('relation_auxiliary',False)
    if has_aux != (ck['condition']=='soft_history_aux') or \
            (has_aux and (ck.get('relation_schema')!=RELATION_SCHEMA or ck.get('relation_weight')!=.1)):
        raise ValueError('checkpoint relation schema/condition mismatch')
    decoder = ChartDecoder(**ck['dec_config'])
    decoder.load_state_dict(ck['model'], strict=True)
    return decoder.to(device).eval()


def forward_loss(decoder, sample, mode, condition, device, *, score=False):
    row = sample['routes'][mode]
    tokens, frames = row['tokens'].to(device), row['frames'].to(device)
    acoustic, valid = row['features'].to(device), sample['valid'].to(device)
    soft = condition_input(sample['soft_onsets'].to(device), condition)
    memory = decoder.prepare_memory(acoustic, soft)
    history = None
    if decoder.history_dim:
        history = history_features(row['tokens'][:-1],row['frames'][:-1],n_lanes=decoder.n_lanes,
                    frames_per_beat=60/row['reference']['timebase_bpm']/MEMORY_SECONDS,
                    offsets=row['offsets'][:-1] if decoder.n_lanes==64 else None)[None].to(device)
    logits, hidden = decoder(tokens[None,:-1], frames[None,:-1], memory, valid, return_hidden=True,history=history)
    ce = F.cross_entropy(logits[0], tokens[1:], ignore_index=PAD)
    offset_loss = logits.new_zeros(())
    mask = row['offset_mask'][:-1].to(device)
    predicted = None
    if decoder.offset_head is not None:
        predicted = decoder.predict_offset(hidden)[0]
        if mask.any():
            offset_loss = (predicted[mask]-row['offsets'][:-1].to(device)[mask]).square().mean()
    details = dict(token_ce_nats=float(ce.detach()), offset_mse=float(offset_loss.detach()),
                   supervised_tokens=len(tokens)-1)
    auxiliary_loss=logits.new_zeros(())
    if decoder.relation_head is not None:
        labels=relation_targets(row['tokens'],row['frames'],n_lanes=decoder.n_lanes).to(device)
        mask_aux=labels!=-100
        if mask_aux.any():
            auxiliary_loss=F.cross_entropy(decoder.relation_head(hidden)[0,mask_aux],labels[mask_aux])
        details['relation_auxiliary']=dict(schema=RELATION_SCHEMA,weight=.1,count=int(mask_aux.sum()),
            ce_nats=float(auxiliary_loss.detach()),
            by_label={name:int((labels==i).sum()) for i,name in enumerate(RELATION_NAMES)})
    if score:
        tk = ChartTokenizer(decoder.n_lanes)
        details['scores'] = token_scores(logits[0], tokens[1:], tk)
        eos = tokens[1:] == EOS
        details['scores']['eos'] = dict(count=int(eos.sum()), bits=float(
            -logits[0].float().log_softmax(-1)[eos, EOS].mean()/math.log(2)) if eos.any() else None)
        details['conditional_offset'] = (offset_scores(predicted,tokens[:-1],row['offsets'][:-1].to(device),mask,tk)
                                         if predicted is not None else None)
    return ce+offset_loss+.1*auxiliary_loss, details


def training_preflight(decoder, sample, mode, budget, device):
    """Conservative decoder math-attention fallback, including backward storage."""
    row = sample['routes'][mode]
    length, frames = len(row['tokens'])-1, row['features'].shape[1]
    depth, heads = len(decoder.blocks), decoder.blocks[0].h
    dim, vocab = decoder.tok_emb.embedding_dim, decoder.lm_head.out_features
    checkpointed = getattr(decoder, 'gradient_checkpointing', False)
    attention_layers = 1 if checkpointed else depth
    parts = dict(attention_and_backward=attention_layers*heads*(length*length+length*frames)*4*4,
                 hidden_and_backward=depth*(length+frames)*dim*4*64,
                 masks=length*frames*12, logits=length*vocab*4*4,
                 optimizer_and_gradient=sum(p.numel()*p.element_size() for p in decoder.parameters())*4)
    extra = sum(parts.values())
    state = budget.check(force=True) or resource_snapshot(budget.directory)
    if device.type == 'cuda':
        free,total = torch.cuda.mem_get_info(device)
        allowance = min(free,total*budget.args.gpu_memory_fraction-torch.cuda.memory_allocated(device))
        if extra > allowance:
            raise RuntimeError('Resource budget: decoder training estimate exceeds GPU budget')
    elif state['rss_gib']+extra/GIB > budget.args.max_rss_gib or \
            state['ram_available_gib']-extra/GIB < budget.args.min_available_ram_gib:
        raise RuntimeError('Resource budget: decoder training estimate exceeds RAM reserve/RSS')
    return dict(method=('float32-block-recompute-forward-backward-v1' if checkpointed
                        else 'float32-attention-forward-backward-v1'), bytes=extra, parts=parts,
                gradient_checkpointing=checkpointed, retained_attention_layers=attention_layers,
                tokens=length, memory_frames=frames)


def weighted_scores(rows):
    out = {}
    for name in rows[0]['scores'] if rows else ():
        count = sum(r['scores'][name]['count'] for r in rows)
        out[name] = dict(count=count,bits=sum(r['scores'][name]['bits']*r['scores'][name]['count']
                         for r in rows if r['scores'][name]['count'])/count if count else None)
    offset = [r['conditional_offset'] for r in rows if r.get('conditional_offset')]
    count = sum(r['count'] for r in offset)
    result=dict(scores=out, conditional_offset=dict(count=count,
                mae_sub_bin=sum(r['mae_sub_bin']*r['count'] for r in offset if r['count'])/count if count else None))
    aux=[r['relation_auxiliary'] for r in rows if r.get('relation_auxiliary')]
    if aux:
        count=sum(r['count'] for r in aux)
        result['relation_auxiliary']=dict(count=count,weight=.1,
            ce_nats=sum(r['ce_nats']*r['count'] for r in aux)/count if count else None,
            by_label={name:sum(r['by_label'][name] for r in aux) for name in RELATION_NAMES})
    return result
