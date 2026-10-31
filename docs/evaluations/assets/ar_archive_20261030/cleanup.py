"""One-shot cleanup after verified publication; keep all latest resume states."""
from datetime import datetime, timezone
import shutil
import torch

from archive import ROOT, DEST, read, write, sha


def main():
    record_path=DEST/'cleanup.json'
    if record_path.exists():
        raise RuntimeError('Cleanup already recorded; inspect its journal instead of repeating it')
    manifest=read(DEST/'manifest.json')
    published=read(DEST/'published.json')
    extra=read(DEST/'supplementary-manifest.json')
    assert published['status']=='verified' and len(published['assets'])==12
    assets={a['name']:a for a in published['assets']}
    protected=[];pending=[];recovery={}
    for run in manifest['runs']:
        r=run['resume'];p=ROOT/r['path'];assert sha(p)==r['sha256']
        full=torch.load(p,map_location='cpu',weights_only=False)
        assert full['epoch']==1400 and full['steps']==420000 and 'training_state' in full
        protected.append({**r,'optimizer_state_keys':list(full['training_state'])})
        del full
        for item in run['snapshots']:
            recovery[item['path']]=dict(url=assets[run['archive']]['url'],
                archive_sha256=assets[run['archive']]['sha256'],
                member=item['path'],sha256=item['sha256'])
    for item in extra['snapshots']:
        recovery[item['path']]=dict(url=assets['AL-AR_additional_e1350.zip']['url'],
            archive_sha256=assets['AL-AR_additional_e1350.zip']['sha256'],
            member=item['path'],sha256=item['sha256'])
    # Require exact tensor equality with a retained/archived inference snapshot.
    for base in ('ar_saturation_20261023','freex_anchored_20261025'):
        for p in sorted((ROOT/'out'/base).rglob('model_latest.pt')):
            if 'continued1400' in p.parts:continue
            assert not p.is_symlink()
            full=torch.load(p,map_location='cpu',weights_only=False)
            candidates=[m for m in p.parent.glob('model_e*.pt')
                        if int(m.stem.removeprefix('model_e'))==full['epoch']]
            assert len(candidates)==1,(p,candidates)
            compact=candidates[0];digest=sha(compact)
            if compact.with_suffix('.json').exists():
                assert read(compact.with_suffix('.json'))['sha256']==digest
            thin=torch.load(compact,map_location='cpu',weights_only=False)
            assert full['model'].keys()==thin['model'].keys()
            assert all(torch.equal(t,thin['model'][k]) for k,t in full['model'].items())
            relative=str(compact.relative_to(ROOT))
            pending.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),
                bytes=p.stat().st_size,epoch=full['epoch'],kind='superseded_full_resume',
                inference_snapshot=relative,
                inference_recovery=recovery.get(relative,dict(local_path=relative,sha256=digest)),
                limitation='Older optimizer/RNG removed; inference parameters preserved; '
                           'exact continuation retained at epoch 1400.'))
            del full,thin
    for relative,remote in recovery.items():
        p=ROOT/relative
        assert p.is_file() and not p.is_symlink() and sha(p)==remote['sha256']
        pending.append(dict(path=relative,sha256=remote['sha256'],bytes=p.stat().st_size,
                            kind='remote_verified_compact',recovery=remote))
    assert len(pending)==35
    assert not {r['path'] for r in protected}.intersection(r['path'] for r in pending)
    record=dict(schema='audiolab-verified-archive-cleanup-v1',
        time_utc=datetime.now(timezone.utc).isoformat(),status='planned',
        disk_free_before=shutil.disk_usage(ROOT).free,protected=protected,files=pending,
        retained='All reports, sidecars, source data, frozen frontends, unarchived snapshots, '
                 'best-fit files, and four latest e1400 complete resume checkpoints.')
    write(record_path,record)
    for row in pending:
        p=ROOT/row['path'];assert sha(p)==row['sha256']
        p.unlink();row['deleted']=True;write(record_path,record)
    for r in protected:assert sha(ROOT/r['path'])==r['sha256']
    record.update(status='completed',freed_bytes=sum(r['bytes'] for r in pending),
                  disk_free_after=shutil.disk_usage(ROOT).free)
    write(record_path,record)
    print(dict(status=record['status'],deleted_files=len(pending),
        archived_snapshots=len(recovery),freed_gib=record['freed_bytes']/2**30,
        disk_free_gib=record['disk_free_after']/2**30,retained_resume=len(protected)))


if __name__=='__main__':main()
