"""Full-song reference-conditioned feature interventions, without training."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import torch

from script.train_models import write_json
from src.audio_understanding.inference import code_version, load_audio, sha256
from src.conditioning_audit import (SCHEMA, CONDITIONS, ReferenceError, file_fingerprint,
    tensor_fingerprint, load_reference, estimate_working_set, enforce_preflight,
    encode_audio, alter_features, score_condition, atomic_report)
from src.data.timing import TIMING_SCHEMA
from src.resource_budget import GIB, ResourceBudget, add_resource_args
from src.test_decoder import load_models


def select_entries(listing, indices):
    listing=Path(listing).resolve(strict=True)
    lines=[(i,line) for i,line in enumerate(listing.read_text().splitlines(),1)
           if line.strip() and not line.lstrip().startswith('#')]
    if not 2<=len(indices)<=3 or len(set(indices))!=len(indices):
        raise ValueError('select two or three distinct development indices')
    entries=[]
    for index in indices:
        if index<0 or index>=len(lines): raise ValueError('data-list index out of range')
        line_number,line=lines[index]
        parts=line.split(',')
        if len(parts)!=3: raise ValueError('data-list row must contain chart,audio,difficulty')
        chart,audio,difficulty=(part.strip() for part in parts)
        entries.append(dict(index=index,list_line=line_number,chart=str((listing.parent/chart).resolve()),
                            audio=str((listing.parent/audio).resolve()),source_difficulty=difficulty))
    return entries


def version():
    result=code_version();root=Path(__file__).resolve().parents[1]
    paths=['src/conditioning_audit.py','script/audit_audio_conditioning_v2.py',
           'script/train_models.py','src/models/chart_decoder.py','src/models/transcriber.py',
           'src/models/attention.py','src/condition/wave.py','src/data/audio2mel.py',
           'src/data/chart_tokenizer.py','src/data/chart2array.py','src/data/timing.py',
           'src/resource_budget.py','src/test_decoder.py']
    result['source_sha256'].update({path:sha256(root/path) for path in paths})
    return result


def error_record(error):
    value=dict(type=type(error).__name__,message=str(error))
    if isinstance(error,ReferenceError): value['details']=error.details
    return value


def checked_models(checkpoint, model):
    metadata=file_fingerprint(checkpoint)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('timing_schema')!=TIMING_SCHEMA:
        raise ValueError('v2 requires a checkpoint with the current explicit timing schema')
    metadata.update(enc_config=ck['enc_config'],dec_config=ck['dec_config'],timing_schema=ck['timing_schema'])
    del ck
    enc,dec=load_models(str(checkpoint),torch.device('cpu'))
    if enc.mel_per_frame!=2 or enc.conv_in.in_channels!=128 or dec.n_lanes!=(4 if model=='decoder' else 64):
        raise ValueError('checkpoint does not match the product layout/feature interface')
    if model=='decoder_freex' and dec.offset_head is None:
        raise ValueError('free-x audit requires the existing continuous offset head')
    if sha256(checkpoint)!=metadata['sha256']:
        raise ValueError('checkpoint changed while loading')
    return enc,dec,metadata


def audit_row(args,entry,other,model,enc,dec,checkpoint_metadata,budget,device):
    row=dict(model=model,input=entry,wrong_input=other,status='running')
    start=time.monotonic()
    try:
        audio=file_fingerprint(entry['audio']);wrong_audio=file_fingerprint(other['audio'])
        row.update(audio=audio,wrong_audio=wrong_audio)
        if audio['sha256']==wrong_audio['sha256']:
            raise ValueError('wrong-song control has identical audio content, even if paths differ')
        mel,duration=load_audio(entry['audio'],max_seconds=args.max_seconds,budget=budget)
        wrong_mel,wrong_duration=load_audio(other['audio'],max_seconds=args.max_seconds,budget=budget)
        gameplay=Path(args.gameplay_dir)/(Path(entry['chart']).parent.name+'.jsonl') if model=='decoder_freex' else None
        reference=load_reference(entry['chart'],duration,gameplay_path=gameplay,source_root=args.source_root)
        row['reference']=reference.metadata
        # Both checks run before full attention, large masks, or token logits.
        row['preflight']=enforce_preflight(estimate_working_set(enc,dec,mel.shape[-1],len(reference.tokens)),budget,device)
        row['wrong_encoder_preflight']=enforce_preflight(estimate_working_set(enc,dec,wrong_mel.shape[-1],2),budget,device)
        enc.to(device);dec.to(device)
        features,valid,feature_meta=encode_audio(enc,mel,duration,device)
        other_features,other_valid,other_meta=encode_audio(enc,wrong_mel,wrong_duration,device)
        row.update(features=feature_meta,wrong_features=other_meta,
                   mel=tensor_fingerprint(mel),wrong_mel=tensor_fingerprint(wrong_mel),conditions={})
        del mel,wrong_mel
        for condition in CONDITIONS:
            budget.check(force=True)
            acoustic,mapping=alter_features(features,valid,condition,other=other_features,other_valid=other_valid)
            value=score_condition(dec,acoustic,valid,reference)
            value.update(mapping=mapping,features=tensor_fingerprint(acoustic),
                         fixed_inputs={**{key:ref['sha256'] for key,ref in reference.metadata['fingerprints'].items()},
                                       'valid_mask':feature_meta['valid_mask']['sha256']})
            row['conditions'][condition]=value
            del acoustic
        row['deltas_from_matched']={condition:{metric:(value['bits']-row['conditions']['matched']['scores'][metric]['bits'])
              if value['bits'] is not None else None for metric,value in row['conditions'][condition]['scores'].items()}
              for condition in CONDITIONS[1:]}
        if model=='decoder_freex':
            row['offset_mae_deltas_from_matched']={condition:
                row['conditions'][condition]['conditional_offset']['mae_sub_bin']-
                row['conditions']['matched']['conditional_offset']['mae_sub_bin'] for condition in CONDITIONS[1:]}
        # Refuse mixed snapshots if external source files changed during the run.
        files=[audio,wrong_audio,checkpoint_metadata,*[v for v in reference.metadata['provenance'].values()
                if isinstance(v,dict) and 'path' in v and 'sha256' in v]]
        if any(sha256(value['path'])!=value['sha256'] for value in files):
            raise ValueError('source input changed during audit')
        row['status']='completed'
    except Exception as error:
        row.update(status='failed',error=error_record(error))
    row.update(elapsed_seconds=time.monotonic()-start,resources=budget.log_state())
    return row


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-list',required=True)
    p.add_argument('--indices',nargs='+',type=int,required=True)
    p.add_argument('--split-label',required=True,help='report provenance; e.g. reused-development-val')
    p.add_argument('--selection-reason',required=True)
    p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--gameplay-dir',required=True)
    p.add_argument('--source-root',required=True,help='explicit base for relative mining source_path')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-seconds',type=float,default=600)
    add_resource_args(p);p.set_defaults(max_rss_gib=4)
    return p


def aggregate_rows(rows):
    summaries={}
    for model in ('decoder','decoder_freex'):
        selected=[row for row in rows if row['model']==model]
        complete=[row for row in selected if row['status']=='completed']
        summary=dict(completed_rows=len(complete),failed_rows=len(selected)-len(complete),
                     scope='completed rows only; token/NOTE-count weighted',conditions={})
        if complete:
            for condition in CONDITIONS:
                metrics={}
                for metric in complete[0]['conditions'][condition]['scores']:
                    values=[row['conditions'][condition]['scores'][metric] for row in complete]
                    count=sum(v['count'] for v in values)
                    metrics[metric]=dict(count=count,bits=sum(v['bits']*v['count'] for v in values if v['count'])/count if count else None)
                summary['conditions'][condition]=dict(scores=metrics)
                if model=='decoder_freex':
                    values=[row['conditions'][condition]['conditional_offset'] for row in complete]
                    count=sum(v['count'] for v in values)
                    summary['conditions'][condition]['conditional_offset']=dict(count=count,
                        mae_sub_bin=sum(v['mae_sub_bin']*v['count'] for v in values)/count,
                        mae_screen_width=sum(v['mae_screen_width']*v['count'] for v in values)/count)
            summary['bits_deltas_from_matched']={condition:{metric:
                value['bits']-summary['conditions']['matched']['scores'][metric]['bits'] if value['bits'] is not None else None
                for metric,value in summary['conditions'][condition]['scores'].items()} for condition in CONDITIONS[1:]}
        summaries[model]=summary
    return summaries


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();device=torch.device(args.device)
    report=dict(schema=SCHEMA,status='running',implementation=version(),config=vars(args),results=[])
    try:
        if not 1<=args.threads<=2: raise ValueError('this bounded audit uses one or two CPU threads')
        torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        budget=ResourceBudget(args,device,output)
        entries=select_entries(args.data_list,args.indices)
        report.update(data_list=file_fingerprint(args.data_list),selection=entries,expected_rows=len(entries)*2,
                      split=args.split_label,selection_reason=args.selection_reason,is_frozen_quality_test=False,
                      runtime=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
                                   device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                                   precision='float32; no autocast; TF32 disabled',seed=args.seed,threads=args.threads),
                      checkpoints={},interpretation='teacher-forced reference likelihood and true-bin offset sensitivity; no generated-playability claim')
        write_json(output/'audit.in_progress.json',report)
        for model in ('decoder','decoder_freex'):
            checkpoint=Path(args.checkpoint_root)/model/'decoder_best.pt'
            try:
                enc,dec,metadata=checked_models(checkpoint,model)
                report['checkpoints'][model]=metadata
            except Exception as error:
                for index,entry in enumerate(entries):
                    report['results'].append(dict(model=model,input=entry,wrong_input=entries[(index+1)%len(entries)],
                                                  status='failed',error=error_record(error)))
                continue
            for index,entry in enumerate(entries):
                row=audit_row(args,entry,entries[(index+1)%len(entries)],model,enc,dec,metadata,budget,device)
                report['results'].append(row)
                write_json(output/'audit.in_progress.json',report)
                print(json.dumps(dict(model=model,index=entry['index'],status=row['status'],
                                      elapsed_seconds=row['elapsed_seconds']),ensure_ascii=False),flush=True)
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            del enc,dec
            gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
        complete=sum(row['status']=='completed' for row in report['results'])
        failed=report['expected_rows']-complete
        report.update(status='completed' if not failed else 'completed_with_failures',completed_rows=complete,
                      failed_rows=failed,elapsed_seconds=time.monotonic()-started,resources=budget.log_state(),
                      model_summaries=aggregate_rows(report['results']),
                      serialization_check='atomic final file then actual JSON readback and invariant validation')
        report['resources']['process_peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB
        budget.check(force=True,extra_bytes=5*1024**2)
        atomic_report(output/'report.json',report)
        return report
    except BaseException as error:
        (output/'report.json').unlink(missing_ok=True)
        failure=dict(schema=SCHEMA,status='failed',error=error_record(error),partial_results=report['results'])
        try: write_json(output/'failure.json',failure)
        except OSError: pass
        raise


if __name__=='__main__':
    result=run(parser().parse_args())
    raise SystemExit(0 if result['status']=='completed' else 1)
