"""H0/H1: raw rollout and paired clean/noisy reference-prefix recovery."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

from collections import Counter
import json
import random
import statistics

import torch

from summarize_controlled import teacher
from src.audio_understanding.inference import sha256
from src.history_time_noise import perturb_history_times
from src.product.artifacts import write_json
from src.windowed_ar import WindowedAR, encode_window_target


def summarize_counts(counts):
    return dict(counts=counts,pointer_accuracy_25ms=counts['pointer_correct_25ms']/max(1,counts['events']),
        continue_and_time_accuracy_25ms=counts['continue_and_time_correct_25ms']/max(1,counts['events']),
        false_stop_rate=counts['false_stops']/max(1,counts['events']),
        stop_recall=counts['correct_stops']/max(1,counts['stops']),
        layout_accuracy=counts['layout_correct']/max(1,counts['layout_total']))


def main():
    torch.set_num_threads(2);base=ROOT/'out/windowed_ar_20261031'
    dataset=torch.load(base/'controlled/conditions.pt',map_location='cpu',weights_only=True)
    rows=[r for r in dataset if r['split'] in ('train','random_check')]
    targets={r['id']:encode_window_target(r['route'],r['duration']) for r in rows}
    result=dict(schema='audiolab-controlled-history-noise-comparison-v1',runs={},noise_seeds=list(range(900,908)),
        scope='Reference-prefix recovery is conditional diagnostics, not free-rollout F1. '
              'No artificial layout labels from the old fixed acoustic probes are used.')
    for name,directory in [('H0','window'),('H1','history20')]:
        root=base/directory;checkpoint=torch.load(root/'model_latest.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==400 and checkpoint['steps']==12800
        model=WindowedAR(checkpoint['config']['decoder_config']).eval();model.load_state_dict(checkpoint['model'])
        evals=json.loads((root/'evaluation-e0400.json').read_text())['rows'];rollouts={}
        for split in sorted({r['split'] for r in evals}):
            subset=[r for r in evals if r['split']==split];positive=[r for r in subset if r['reference_onsets']]
            counts={key:sum(r['onset']['25'][key] for r in subset) for key in ('reference','predicted','matched')}
            summary=dict(examples=len(subset),positive_examples=len(positive),completed=sum(r['status']=='completed' for r in subset),
                positive_macro_raw_onset_f1_25ms=statistics.fmean(r['onset']['25']['f1'] for r in positive),
                raw_onset_micro={**counts,'precision':counts['matched']/max(1,counts['predicted']),'recall':counts['matched']/max(1,counts['reference'])},
                silence_extra_onsets=sum(len(r['onsets_seconds']) for r in subset if not r['reference_onsets']),
                forced_windows=sum(r['forced_windows'] for r in subset))
            if split!='fixed_probe':summary['positive_macro_joint_f1_25ms']=statistics.fmean(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in positive)
            rollouts[split]=summary
        recovery={}
        for split in ('train','random_check'):
            subset=[r for r in rows if r['split']==split];clean=Counter();noisy=Counter();audit=Counter();per_seed=[]
            for row in subset:clean.update(teacher(model,targets[row['id']],row['condition'],True))
            for seed in result['noise_seeds']:
                stats=Counter();rng=random.Random(seed)
                for row in subset:
                    t,a=perturb_history_times(targets[row['id']],n_lanes=4,max_seconds=.02,rng=rng)
                    stats.update(teacher(model,t,row['condition'],True));audit.update({k:v for k,v in a.items() if k!='max_absolute_noise_seconds'})
                noisy.update(stats);per_seed.append(dict(seed=seed,**summarize_counts(stats)))
            recovery[split]=dict(clean=summarize_counts(clean),noisy=summarize_counts(noisy),per_seed=per_seed,noise_audit=audit)
        training=json.loads((root/'training.json').read_text())
        result['runs'][name]=dict(checkpoint=str((root/'model_latest.pt').relative_to(ROOT)),
            checkpoint_sha256=sha256(root/'model_latest.pt'),evaluation_sha256=sha256(root/'evaluation-e0400.json'),
            config=checkpoint['config'],rollout=rollouts,recovery=recovery,
            final_training=training['curves'][-1],resources=training['resources'])
        print(json.dumps(dict(arm=name,rollout=rollouts,recovery={s:{k:v for k,v in r.items() if k!='per_seed'} for s,r in recovery.items()})),flush=True)
    path=ROOT/'docs/evaluations/assets/windowed_ar_20261031/h0_h1_controlled_result.json'
    write_json(path,result)


if __name__=='__main__':main()
