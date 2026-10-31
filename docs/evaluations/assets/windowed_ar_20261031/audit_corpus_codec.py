"""Bounded all300 reference audit; does not train or evaluate a generator."""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

import json
import time
from types import SimpleNamespace
import torch
from script.train_soft_onset import cache_inputs, load_sample
from src.anchored_ar import encode_target
from src.windowed_ar import encode_window_target, window_intervals, window_time_bins
from src.data.chart_tokenizer import ChartTokenizer
from src.audio_understanding.inference import sha256
from src.resource_budget import ResourceBudget
from src.product.artifacts import write_json
from src.soft_onset import MEMORY_SECONDS


def main():
    torch.set_num_threads(2);root=ROOT/'out/full_corpus_v1/cache';cache=cache_inputs(root)
    started=time.monotonic();rows=[];groups=coordinates=0;error=0.;eow=ChartTokenizer(64).cond(2)
    args=SimpleNamespace(min_free_disk_gib=8,min_available_ram_gib=8,max_rss_gib=4,gpu_memory_fraction=.65,keep_checkpoints=1)
    budget=ResourceBudget(args,torch.device('cpu'),ROOT/'out/windowed_ar_20261031')
    for row in cache['rows']:
        budget.check();sample=load_sample(root,row);route=sample['routes']['freex'];duration=sample['duration']
        a=encode_target(route,duration,n_lanes=64);b=encode_window_target(route,duration,n_lanes=64)
        assert torch.equal(b['tokens'][b['tokens']!=eow],a['tokens'])
        assert torch.equal(a['offsets'][a['offset_mask']],b['offsets'][b['offset_mask']])
        assert a['time_count']==b['time_count'] and a['coordinate_count']==b['coordinate_count']
        positives=torch.where(b['stop_targets']==0)[0];length=sample['valid'].shape[-1]
        for chunk in positives.split(64):
            indexes=b['decisions'][chunk];truth=b['next_seconds'][chunk]
            lo,hi,allowed=window_intervals(b['lower_seconds'][chunk],b['bounds'][indexes,1],length,sample['valid'])
            bins=window_time_bins(truth,length);r=torch.arange(len(chunk));assert allowed[r,bins].all()
            f=(truth-lo[r,bins])/(hi[r,bins]-lo[r,bins])
            bad=torch.where((f<0)|(f>=1+1e-9))[0]
            if len(bad):
                i=int(bad[0])
                raise ValueError(json.dumps(dict(song=row['id'],truth=float(truth[i]),bin=int(bins[i]),
                    lo=float(lo[i,bins[i]]),hi=float(hi[i,bins[i]]),fraction=float(f[i]),
                    lower=float(b['lower_seconds'][chunk[i]]),decision=int(indexes[i]))))
            if len(chunk):error=max(error,float((lo[r,bins]+f*(hi[r,bins]-lo[r,bins])-truth).abs().max()))
        groups+=a['time_count'];coordinates+=a['coordinate_count']
        rows.append(dict(id=row['id'],groups=a['time_count'],notes=a['coordinate_count'],windows=b['stop_count']))
        del sample,route,a,b
        if len(rows)%100==0:print(json.dumps(dict(audited=len(rows),groups=groups,notes=coordinates,resources=budget.log_state())),flush=True)
    result=dict(scope='All 300 original free-x references; codec only, no new training or quality evaluation.',
        data_cache_sha256=sha256(root/'cache.json'),groups=groups,coordinates=coordinates,whole_stop_labels=len(rows),
        window_stop_labels=sum(r['windows'] for r in rows),max_time_roundtrip_error_seconds=error,
        original_residuals_equal=True,elapsed_seconds=time.monotonic()-started,rows=rows,resources=budget.log_state())
    write_json(ROOT/'docs/evaluations/assets/windowed_ar_20261031/freex_corpus_codec_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
