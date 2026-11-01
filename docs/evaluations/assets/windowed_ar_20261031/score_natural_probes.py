"""Original isolated-onset probes on the natural parent and adapted models."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

import json
import torch
from script.train_windowed_ar import score
from src.anchored_ar import AnchoredAR, generate as whole_generate
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.windowed_ar import WindowedAR, generate as window_generate


@torch.no_grad()
def main():
    torch.set_num_threads(2);base=ROOT/'out/windowed_ar_20261031'
    rows=[r for r in torch.load(base/'controlled/conditions.pt',map_location='cpu',weights_only=True) if r['split']=='fixed_probe']
    checkpoints=[('parent_A1400',ROOT/'out/ar_saturation_20261023/continued1400/audio/model_latest.pt','whole'),
        ('NW0',base/'natural_whole/model_latest.pt','whole'),('NW1',base/'natural_window/model_latest.pt','window')]
    result={}
    for name,path,architecture in checkpoints:
        if name!='parent_A1400':
            t=json.loads(path.with_name('training.json').read_text())
            if t['status']!='completed':raise RuntimeError('finish natural pilot before evaluating its checkpoint')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model=(WindowedAR if architecture=='window' else AnchoredAR)(ck['config']['decoder_config']).eval()
        model.load_state_dict(ck['model'],strict=True);del ck
        outputs=[]
        for row in rows:
            p=(window_generate if architecture=='window' else whole_generate)(model,row['condition'],120.,
                use_soft=False,greedy=True,max_tokens=512)
            p.pop('trace',None);score(p,row);outputs.append(dict(id=row['id'],**p))
        positive=[r for r in outputs if r['reference_onsets']]
        summary=dict(completed=sum(r['status']=='completed' for r in outputs),examples=len(outputs),
            positive_onset_f1_25ms=sum(r['onset']['25']['f1'] for r in positive)/len(positive),
            silence_extra_onsets=sum(r['generated_onsets'] for r in outputs if not r['reference_onsets']))
        result[name]=dict(checkpoint_sha256=sha256(path),summary=summary,rows=outputs)
        print(json.dumps(dict(run=name,**summary)),flush=True)
    write_json(Path(__file__).with_name('natural_fixed_probes.json'),dict(schema='audiolab-natural-window-fixed-probes-v1',
        runs=result,data_sha256=sha256(base/'controlled/conditions.pt'),script_sha256=sha256(__file__),
        device='cpu',greedy=True,max_tokens=512,training_updates=0,
        scope='Original five fixed acoustic probes, not in natural training. Dummy type/position labels excluded. '
            'Same CPU inference and 512-token diagnostic budget in all arms; not the natural-song 12000-token limit.'))


if __name__=='__main__':main()
