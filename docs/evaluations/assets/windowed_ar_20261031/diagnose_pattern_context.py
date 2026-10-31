"""Read-only context dependence probes; NOT a no-summary retraining result."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

from collections import defaultdict
import json
import torch

from script.train_pattern_continuation import continuation, PREFIX
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.windowed_variants import VariantAR

BASE=ROOT/'out/windowed_ar_20261031'


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    checkpoint=BASE/'pattern_base/model_latest.pt';data_path=BASE/'patterns/conditions.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ck['epoch']==400 and ck['config']['data_sha256']==sha256(data_path)
    model=VariantAR(ck['config']['decoder_config']).eval();model.load_state_dict(ck['model'])
    rows=torch.load(data_path,map_location='cpu',weights_only=True)['rows']
    original=model.decoder.step;result={};predictions={}
    for name,zero_summary,current_only in [('full',False,False),('zero_summary',True,False),
        ('current_token_only',False,True),('both_removed',True,True)]:
        def step(*args,**kwargs):
            if zero_summary:kwargs['history']=torch.zeros_like(kwargs['history'])
            if current_only:
                for cache in args[3]:cache['k']=cache['v']=None
            return original(*args,**kwargs)
        model.decoder.step=step;totals=defaultdict(lambda:dict(examples=0,correct=0,positions=0,exact=0));outputs=[]
        try:
            for row in rows:
                pred=continuation(model,row['condition'],row['times'],row['prefix'])
                correct=sum(a==b for a,b in zip(pred[PREFIX:],row['lanes'][PREFIX:]))
                for key in (row['split'],row['split']+'/'+row['family']):
                    total=totals[key];total['examples']+=1;total['correct']+=correct
                    total['positions']+=len(pred)-PREFIX;total['exact']+=pred==row['lanes']
                outputs.append(dict(id=row['id'],suffix=pred[PREFIX:],correct=correct))
        finally:model.decoder.step=original
        result[name]={k:dict(v,accuracy=v['correct']/v['positions']) for k,v in totals.items()}
        predictions[name]=outputs
        print(json.dumps(dict(intervention=name,train=result[name]['train'],new_combinations=result[name]['new_combinations'])),flush=True)
    # All inference variants use CPU arithmetic. Establish the unchanged control
    # before interpreting differences from the GPU-trained model's saved output.
    reference=json.loads((BASE/'pattern_base/evaluation-e0400.json').read_text())
    by_id={r['id']:r['predicted_suffix'] for r in reference['rows']}
    assert all(r['suffix']==by_id[r['id']] for r in predictions['full'])
    output=dict(schema='audiolab-pattern-context-dependence-v1',checkpoint_sha256=sha256(checkpoint),
        data_sha256=sha256(data_path),script_sha256=sha256(__file__),metrics=result,predictions=predictions,
        original_cpu_gpu_rollout_equal=True,training_updates=0,
        interpretation='Inference interventions on a model trained WITH summary and causal self-history. '
            'Distribution changes measure dependence; they do not establish the capability of retrained alternatives. '
            'Whole audio, clock, window bounds, eight observed prefix notes and one-Tap-per-given-onset contract remain.')
    write_json(Path(__file__).with_name('pattern_context_dependence.json'),output)


if __name__=='__main__':main()
