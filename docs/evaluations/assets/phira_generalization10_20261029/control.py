"""Re-encode one training song to distinguish new-song behavior from a broken frontend."""
from pathlib import Path
from types import SimpleNamespace
import gc
import importlib.util
import json
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
sys.path.insert(0,str(ROOT))
import torch
from script.train_grounded_ar import load_inputs, encode_waveform
from src.anchored_ar import AnchoredAR, generate
from src.data.audio2mel import AudioGPUprocessor
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget


def main():
    torch.set_num_threads(2)
    assets=Path(__file__).resolve().parent
    plan=json.loads((assets/'selection.json').read_text())
    worker_path=ROOT/'docs/evaluations/assets/ar_continue1200_20261027/worker.py'
    spec=importlib.util.spec_from_file_location('fixed15_worker',worker_path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    cache=json.loads((ROOT/'out/full_corpus_v1/cache/cache.json').read_text())
    row=next(r for r in cache['rows'] if r['id']=='Eltaw_IN')
    sample=torch.load(ROOT/'out/full_corpus_v1/cache'/row['file'],map_location='cpu',weights_only=True)
    waveform=AudioGPUprocessor(device='cpu').load_from_path(row['audio']['path'])
    args=SimpleNamespace(min_free_disk_gib=8,min_available_ram_gib=8,max_rss_gib=4,
        gpu_memory_fraction=.65,keep_checkpoints=1,model_root=ROOT/'checkpoints/timing_v1',mode='4k')
    device=torch.device('cuda:0');budget=ResourceBudget(args,device,ROOT/'out/phira_generalization10_20261029')
    results=[]
    for mode in ('4k','freex'):
        args.mode=mode;encoder,onset=load_inputs(args,device)
        view=encode_waveform(waveform,encoder,onset,budget,device)
        actual=view['features'].float();cached=sample['routes'][mode]['features'].float()
        difference=(actual-cached).abs()
        soft_diff=(view['soft_onsets']-sample['soft_onsets']).abs()
        feature_check=dict(shape=list(actual.shape),duration=view['duration'],cached_duration=sample['duration'],
            relative_l2=float(difference.norm()/cached.norm()),max_abs=float(difference.max()),
            soft_mean_abs=float(soft_diff.mean()),soft_max_abs=float(soft_diff.max()))
        truth,clock=worker.reference(sample,mode)
        for candidate in plan['candidates']:
            if candidate['mode']!=mode:continue
            ck=torch.load(ROOT/candidate['checkpoint'],map_location='cpu',weights_only=True)
            model=AnchoredAR(ck['config']['decoder_config']).to(device).eval()
            model.load_state_dict(ck['model'],strict=True);del ck
            with torch.inference_mode():
                pred=generate(model,view,clock['timebase_bpm'],use_soft=candidate['variant']=='soft',greedy=True,seed=42,check=budget.check)
            metrics=worker.score(truth,pred,clock,plan['rhythm_protocol'])
            previous=json.loads((ROOT/Path(candidate['checkpoint']).parent/f'evaluation-e{candidate["epoch"]}/fixed15/summary.json').read_text())
            saved=next(r for r in previous['songs'] if r['song']=='Eltaw_IN')
            keys=['onset_f1','joint_time_bin_type_f1','grid_bonus']
            item=dict(id=candidate['id'],epoch=candidate['epoch'],features=feature_check,
                status=pred['status'],metrics={k:metrics[k] for k in keys},
                cached_status=saved['status'],cached_metrics={k:saved['metrics'][k] for k in keys})
            results.append(item);print(json.dumps(item),flush=True)
            del pred,model;gc.collect();torch.cuda.empty_cache()
        del view,encoder,onset;gc.collect();torch.cuda.empty_cache()
    write_json(assets/'rollout_control.json',dict(song='Eltaw_IN',split='training control only; excluded from new-song aggregate',
        purpose='same checkpoint and live audio frontend versus stored fixed15 training-song evaluation',results=results,resources=budget.log_state()))


if __name__=='__main__':main()
