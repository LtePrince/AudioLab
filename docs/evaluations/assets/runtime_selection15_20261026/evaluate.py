"""Fixed-seed 15-song release selection; cached audio, no training or repairs."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse, ast, gc, hashlib, json, random, statistics, sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
import torch
from src.anchored_ar import AnchoredAR, generate, residuals_by_note
from src.audio_understanding.evaluation import match_events
from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
from src.pattern_diagnostics import KINDS, LANE_X
from src.model_benchmark import rhythm_bonus
from src.resource_budget import ResourceBudget
from src.product.artifacts import write_json

torch.set_num_threads(2)
DEST = ROOT/'out/runtime_delivery_20261026'
CACHE = ROOT/'out/full_corpus_v1/cache'
METRIC = ROOT/'docs/evaluations/assets/freex_layout_best_20261026/analyze.py'
tree = ast.parse(METRIC.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('groups','compare')],type_ignores=[]),str(METRIC),'exec'))
manifest=json.loads((CACHE/'cache.json').read_text())
protocol=json.loads((ROOT/'benchmarks/v1/protocol.json').read_text())

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def reference(sample,mode):
    route=sample['routes'][mode];ref=route['reference'];scale=60/(32*ref['timebase_bpm'])
    offsets=residuals_by_note(route) if mode=='freex' else None
    notes=[dict(time_seconds=n.tick*scale+ref['offset_seconds'],end_seconds=(n.tick+n.dur)*scale+ref['offset_seconds'],position=n.lane,type=KINDS[n.type],screen_x=bin_to_hit(n.lane,offsets[n.tick,n.lane]) if offsets is not None else LANE_X[n.lane]) for n in ChartTokenizer(64 if mode=='freex' else 4).decode_tokens(route['tokens'].tolist(),strict=True)]
    return notes,ref

def metrics(ref,pred,bpm):
    m,_,_=compare(ref,pred['notes'])
    # Grid scoring uses the exact original chart clock, not a new fitted phase.
    m['generation_status']=pred['status']
    m['selection_joint_f1']=m['joint_time_bin_type_f1'] if pred['status']=='completed' else 0.
    return m

def prepare():
    assert not (DEST/'selection.json').exists()
    pool=sorted((r for r in manifest['rows'] if r['id']!='Eltaw_IN'),key=lambda r:r['id'])
    chosen=[next(r for r in manifest['rows'] if r['id']=='Eltaw_IN')]+random.Random(20261026).sample(pool,14)
    source_refs={}
    for row in manifest['rows']:
        if row['id'] not in ('Eltaw_IN','-SURREALISM-_IN','ABYSS MOTION 192.333_IN'):continue
        s=torch.load(CACHE/row['file'],map_location='cpu',weights_only=True)
        for mode in ('4k','freex'):source_refs[mode,row['id']]=reference(s,mode)[0]
        del s
    all_candidates=[];shortlist={}
    for mode,base in (('4k',ROOT/'out/ar_saturation_20261023'),('freex',ROOT/'out/freex_anchored_20261025/continued1000')):
        ranked=[]
        for path in sorted(base.glob('**/evaluation-e*/inputs.json')):
            epoch=int(path.parent.name.removeprefix('evaluation-e'))
            snapshot=path.parent.parent/f'model_e{epoch:04d}.pt'
            if not snapshot.exists():continue
            data=json.loads(path.read_text())
            for run in data['runs']:
                variant=run['metadata'].get('variant') or ('soft' if 'soft' in run['id'] else 'audio')
                decoding=run['id'].rsplit('/',1)[-1]
                if variant not in ('audio','soft') or decoding not in ('greedy','seed42'):continue
                scores=[]
                for name in ('Eltaw_IN','-SURREALISM-_IN','ABYSS MOTION 192.333_IN'):
                    p=run['predictions'][name];m=metrics(source_refs[mode,name],p,None);scores.append(m)
                ranked.append(dict(mode=mode,variant=variant,epoch=epoch,decoding=decoding,checkpoint=str(snapshot.relative_to(ROOT)),screen_mean_joint=statistics.mean(s['selection_joint_f1'] for s in scores),screen_completed=sum(s['generation_status']=='completed' for s in scores),screen_mean_onset=statistics.mean(s['onset_f1'] for s in scores),source=str(path.relative_to(ROOT))))
        ranked.sort(key=lambda r:(r['screen_mean_joint'],r['screen_completed'],r['screen_mean_onset'],r['epoch']),reverse=True)
        picks=[]
        for variant in ('audio','soft'):
            picks.append(next(r for r in ranked if r['variant']==variant))
        picks.append(next(r for r in ranked if r['checkpoint'] not in {x['checkpoint'] for x in picks}))
        for i,pick in enumerate(picks):
            pick['candidate_id']=f'{mode}-{pick["variant"]}-e{pick["epoch"]:04d}-{pick["decoding"]}'
            pick['sha256']=sha(ROOT/pick['checkpoint'])
        shortlist[mode]=picks;all_candidates.extend(ranked)
    result=dict(schema='audiolab-release-selection15-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),seed=20261026,
        scope='Eltaw + 14 unique songs drawn uniformly without replacement from the existing 300 training songs; not held-out generalization.',
        selection='Screen retained E14/E18 checkpoints using existing three-song macro time+position-bin+type F1, failed generations score zero; retain best A, best S, and highest-ranked remaining distinct checkpoint. Select final mode winner by the same failed-as-zero macro F1 over fixed 15 songs. Greedy/seed42 setting is part of the candidate, not tuned on the 15.',
        cache_sha256=sha(CACHE/'cache.json'),metric_sha256=sha(METRIC),songs=chosen,candidates=shortlist,screening=all_candidates)
    write_json(DEST/'selection.json',result)
    print('SONGS',json.dumps([r['id'] for r in chosen],ensure_ascii=False),flush=True)
    print('CANDIDATES',json.dumps(shortlist,ensure_ascii=False),flush=True)

def run(mode,device):
    plan=json.loads((DEST/'selection.json').read_text());device=torch.device(device)
    budget=ResourceBudget(SimpleNamespace(min_free_disk_gib=8,min_available_ram_gib=8,max_rss_gib=4,gpu_memory_fraction=.65,keep_checkpoints=1),device,DEST)
    for candidate in plan['candidates'][mode]:
        checkpoint=ROOT/candidate['checkpoint'];assert sha(checkpoint)==candidate['sha256']
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['epoch']==candidate['epoch'] and ck['config']['variant']==candidate['variant']
        model=AnchoredAR(ck['config']['decoder_config']).to(device).eval();model.load_state_dict(ck['model'],strict=True);del ck
        folder=DEST/'evaluation'/candidate['candidate_id'];folder.mkdir(parents=True,exist_ok=True)
        rows=[]
        for index,row in enumerate(plan['songs']):
            path=folder/f'{index:02d}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['song']==row['id'] and old['checkpoint_sha256']==candidate['sha256'];rows.append(old);continue
            budget.check(force=True);cache_path=CACHE/row['file'];assert sha(cache_path)==row['sha256']
            sample=torch.load(cache_path,map_location='cpu',weights_only=True)
            ref,metadata=reference(sample,mode)
            view=dict(features=sample['routes'][mode]['features'],soft_onsets=sample['soft_onsets'],valid=sample['valid'],duration=sample['duration'])
            with torch.inference_mode():pred=generate(model,view,metadata['timebase_bpm'],use_soft=candidate['variant']=='soft',greedy=candidate['decoding']=='greedy',seed=42,check=budget.check)
            pred.pop('trace',None)
            score=metrics(ref,pred,metadata['timebase_bpm'])
            result=dict(song=row['id'],mode=mode,candidate_id=candidate['candidate_id'],checkpoint_sha256=candidate['sha256'],audio_sha256=row['audio']['sha256'],reference=metadata,duration=sample['duration'],metrics=score,prediction=pred)
            write_json(path,result);rows.append(result)
            write_json(DEST/f'progress-{mode}.json',dict(candidate=candidate['candidate_id'],songs_done=index+1,total=15,last_song=row['id'],status=pred['status'],resources=budget.log_state()))
            print(json.dumps(dict(candidate=candidate['candidate_id'],song=row['id'],done=index+1,status=pred['status'],onset=score['onset_f1'],layout=score['joint_time_bin_type_f1']),ensure_ascii=False),flush=True)
            del sample,ref,view;gc.collect()
        summary=dict(candidate=candidate,mean_joint_f1=statistics.mean(r['metrics']['selection_joint_f1'] for r in rows),mean_onset_f1=statistics.mean(r['metrics']['onset_f1'] if r['prediction']['status']=='completed' else 0 for r in rows),completed=sum(r['prediction']['status']=='completed' for r in rows),total=15)
        write_json(folder/'summary.json',summary);print('CANDIDATE_DONE',json.dumps(summary),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(DEST/f'progress-{mode}.json',dict(status='completed',resources=budget.log_state()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('--mode',choices=('4k','freex'));p.add_argument('--device',default='cuda:0');args=p.parse_args()
    prepare() if args.action=='prepare' else run(args.mode,args.device)
