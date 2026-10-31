"""Explicit monotonic modulo-4 staircases, beyond arbitrary four-lane cycles."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

import json
import torch
from script.train_pattern_continuation import continuation, PREFIX
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.windowed_variants import VariantAR


@torch.no_grad()
def main():
    torch.set_num_threads(2);base=ROOT/'out/windowed_ar_20261031'
    data=torch.load(base/'patterns/conditions.pt',map_location='cpu',weights_only=True)
    audio=[r for r in data['rows'] if r['split']=='new_combinations' and r['family']=='staircase']
    results={}
    for name,folder,variant in [('P0','pattern_base','base'),('P1','pattern_kind_position','kind_position')]:
        ck=torch.load(base/folder/'model_latest.pt',map_location='cpu',weights_only=True)
        model=VariantAR(ck['config']['decoder_config'],variant=variant).eval();model.load_state_dict(ck['model'])
        rows=[]
        for row in audio:
            for direction in (-1,1):
                for start in range(4):
                    expected=[(start+direction*i)%4 for i in range(24)]
                    pred=continuation(model,row['condition'],row['times'],expected[:PREFIX])
                    rows.append(dict(audio_group=row['audio_group'],direction=direction,start=start,
                        expected=expected[PREFIX:],prediction=pred[PREFIX:],exact=pred==expected,
                        correct=sum(a==b for a,b in zip(pred[PREFIX:],expected[PREFIX:]))))
        results[name]=dict(examples=len(rows),exact=sum(r['exact'] for r in rows),
            position_accuracy=sum(r['correct'] for r in rows)/(16*len(rows)),rows=rows,
            checkpoint_sha256=sha256(base/folder/'model_latest.pt'))
        print(json.dumps(dict(run=name,**{k:v for k,v in results[name].items() if k not in ('rows','checkpoint_sha256')})),flush=True)
    write_json(Path(__file__).with_name('standard_stairs.json'),dict(schema='audiolab-modulo4-stairs-v1',results=results,
        data_sha256=sha256(base/'patterns/conditions.pt'),script_sha256=sha256(__file__),training_updates=0,
        definition='Eight standard 4-lane cycles: start in any lane, then move +1 or -1 modulo four. '
            'Eight new-combination audio waveforms, given times and first eight Tap positions. '
            'A stricter supplemental check: original staircase family includes arbitrary lane permutations.'))


if __name__=='__main__':main()
