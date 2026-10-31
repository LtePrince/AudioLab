"""Archive existing AR milestones, verify remote bytes, then prune redundancies.

Invoked explicitly for the user's 2026-10-30 backup request. No new weights,
training, model conversion, or published runtime bundle replacement.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
DEST=Path(__file__).resolve().parent
WORK=ROOT/'out/ar_archive_20261030'
REPO='LtePrince/AudioLab'
TAG='ar-milestones-20261030'
RESERVE=8*1024**3


def read(p):return json.loads(Path(p).read_text())


def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def command(*args,env=None):
    p=subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.strip())
    return p.stdout


def credentials():
    p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        cwd=ROOT,text=True,capture_output=True,check=True)
    fields=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    env=os.environ.copy();env['GH_TOKEN']=fields['password'];return env


def release(env):
    return json.loads(command('gh','api',f'repos/{REPO}/releases/tags/{TAG}',env=env))


def verify_asset(path,env):
    info=release(env);a=next(a for a in info['assets'] if a['name']==path.name)
    digest=sha(path);assert a['size']==path.stat().st_size
    if a.get('digest'):
        assert a['digest']=='sha256:'+digest
    else:
        p=subprocess.Popen(['gh','api',f'repos/{REPO}/releases/assets/{a["id"]}',
            '-H','Accept: application/octet-stream'],env=env,cwd=ROOT,stdout=subprocess.PIPE)
        received=hashlib.file_digest(p.stdout,'sha256').hexdigest()
        assert p.wait()==0 and received==digest
    return dict(name=a['name'],bytes=a['size'],sha256=digest,url=a['browser_download_url'],id=a['id'])


def upload(path,env):
    if not any(a['name']==path.name for a in release(env)['assets']):
        command('gh','release','upload',TAG,str(path),'--repo',REPO,env=env)
    return verify_asset(path,env)


def plan():
    runs=[]
    for mode,base,product in [('4k','out/ar_saturation_20261023','AL-AR4K_v0.5.1'),
                             ('freex','out/freex_anchored_20261025','AL-FreeX_v0.4.0')]:
        for variant,condition in [('audio','A'),('soft','S')]:
            rows=[]
            epochs=[1100,1200,1300,1400]
            if mode=='freex' and condition=='A':epochs.insert(1,1150)
            for epoch in epochs:
                stage=1200 if epoch<=1200 else 1400
                p=ROOT/base/f'continued{stage}'/variant/f'model_e{epoch}.pt'
                meta=read(p.with_suffix('.json'));assert meta['sha256']==sha(p) and meta['epoch']==epoch
                summary=p.parent/f'evaluation-e{epoch}/fixed15/summary.json';metrics=read(summary)
                assert metrics['checkpoint_sha256']==meta['source_checkpoint_sha256']
                rows.append(dict(epoch=epoch,path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                    sha256=meta['sha256'],full_checkpoint_sha256=meta['source_checkpoint_sha256'],
                    fixed15_train_joint_f1=metrics['mean_joint_f1'],fixed15_completed=metrics['completed']))
            latest=ROOT/base/'continued1400'/variant/'model_latest.pt'
            assert sha(latest)==rows[-1]['full_checkpoint_sha256']
            runs.append(dict(mode=mode,variant=variant,condition=condition,product=product,
                archive=f'{product}_{condition}_e1100-e1400.zip',snapshots=rows,
                resume=dict(path=str(latest.relative_to(ROOT)),epoch=1400,bytes=latest.stat().st_size,
                    sha256=sha(latest),asset=f'{product}_{condition}_resume_e1400.pt')))
    return dict(schema='audiolab-ar-milestone-archive-v1',tag=TAG,
        source_commit=command('git','rev-parse','HEAD').strip(),runs=runs,
        quality='experimental; training-song fit improved, new-song generalization not passed',
        contents='original inference snapshots plus latest full AdamW/RNG resume checkpoints; no audio/charts/frontends',
        published_runtime_bundles_changed=False)


def publish():
    WORK.mkdir(exist_ok=True)
    manifest=read(DEST/'manifest.json') if (DEST/'manifest.json').exists() else plan()
    write(DEST/'manifest.json',manifest)
    env=credentials();assets=read(DEST/'published.json')['assets'] if (DEST/'published.json').exists() else []
    notes=DEST/'release-notes.md'
    if not notes.exists():
        notes.write_text('''# AR 关键轮次与续训断点存档

保留 2026-10-26 运行包发布后的关键训练权重，供恢复与对照；不是新模型训练或质量认证。

- 四组：AL-AR4K v0.5.1 A/S、AL-FreeX v0.4.0 A/S，各 1100/1200/1300/1400 轮。
- free-x A 额外保留 1150 轮阶段最佳；新歌 E20 采用 A1300，其余三组采用 1400。
- 四个 ZIP 保存原始紧凑检查点，按仓库相对路径解压；不包含音频、谱包或冻结前端。
- 四个 resume_e1400.pt 是最新完整断点，包含 AdamW、采样器与 Torch/CUDA RNG，保留续训能力。
- manifest.json 记录每份原文件 SHA256、轮次和训练曲成绩；SHA256SUMS.txt 校验本 Release 附件。

这些是研究恢复文件，不能直接交给 script.model_bundle install。浏览器一键运行仍使用 anchored-ar-20261026 的原 B03/B04 安装包；没有替换已发布运行权重。

训练曲拟合不代表泛化通过：E20 十首新歌仍有时间跳转、段落缺音与停止问题。归因报告位于 docs/evaluations/phira_gap_diagnosis_20261030.md。
''')
    # Build one ZIP at a time, verify it, upload it, then release temporary disk.
    for run in manifest['runs']:
        path=WORK/run['archive']
        if not path.exists():
            required=sum(x['bytes'] for x in run['snapshots'])+2*1024**2
            if shutil.disk_usage(WORK).free<RESERVE+required:raise RuntimeError('8 GiB disk reserve would be crossed')
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for item in run['snapshots']:
                    source=ROOT/item['path'];assert sha(source)==item['sha256']
                    z.write(source,item['path']);z.write(source.with_suffix('.json'),str(Path(item['path']).with_suffix('.json')))
                z.writestr('archive-manifest.json',json.dumps(run,ensure_ascii=False,indent=2)+'\n')
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for item in run['snapshots']:
                with z.open(item['path']) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
        p=subprocess.run(['gh','api',f'repos/{REPO}/releases/tags/{TAG}'],env=env,cwd=ROOT,text=True,capture_output=True)
        if p.returncode:
            if '404' not in p.stderr:raise RuntimeError(p.stderr)
            command('gh','release','create',TAG,'--repo',REPO,'--target',manifest['source_commit'],
                '--prerelease','--title','AudioLab AR 关键轮次存档：1100–1400＋完整续训断点',
                '--notes-file',str(notes),env=env)
        verified=upload(path,env)
        assets=[a for a in assets if a['name']!=verified['name']]+[verified]
        write(DEST/'published.json',dict(tag=TAG,url=release(env)['html_url'],assets=assets,status='uploading'))
        path.unlink()
        latest=ROOT/run['resume']['path'];alias=WORK/run['resume']['asset']
        assert sha(latest)==run['resume']['sha256']
        if not alias.exists():alias.symlink_to(latest)
        verified=upload(alias,env);alias.unlink()
        assets=[a for a in assets if a['name']!=verified['name']]+[verified]
        write(DEST/'published.json',dict(tag=TAG,url=release(env)['html_url'],assets=assets,status='uploading'))
        print(json.dumps(dict(archived=run['product']+'/'+run['condition'],epochs=[x['epoch'] for x in run['snapshots']],verified_assets=len(assets),disk_free_gib=shutil.disk_usage(WORK).free/2**30)),flush=True)
    asset=upload(DEST/'manifest.json',env);assets=[a for a in assets if a['name']!=asset['name']]+[asset]
    checksums=DEST/'SHA256SUMS.txt';checksums.write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in sorted(assets,key=lambda x:x['name'])))
    check=upload(checksums,env);assets=[a for a in assets if a['name']!=check['name']]+[check]
    final=release(env);assert len(final['assets'])==10 and final['prerelease'] and not final['draft']
    write(DEST/'published.json',dict(tag=TAG,url=final['html_url'],assets=assets,status='verified',
        resume_checkpoints_retained_locally=[r['resume'] for r in manifest['runs']]))
    print(json.dumps(dict(status='verified',url=final['html_url'],assets=len(assets)),ensure_ascii=False),flush=True)


def supplement():
    """Add two useful metric trade-offs without replacing verified base assets."""
    env=credentials()
    published=read(DEST/'published.json')
    assert published['status']=='verified'
    rows=[]
    for product,base,variant,condition in [
        ('AL-AR4K_v0.5.1','ar_saturation_20261023','soft','S'),
        ('AL-FreeX_v0.4.0','freex_anchored_20261025','audio','A'),
    ]:
        source=ROOT/'out'/base/'continued1400'/variant/'model_e1350.pt'
        metadata=read(source.with_suffix('.json'))
        summary=read(source.parent/'evaluation-e1350/fixed15/summary.json')
        assert sha(source)==metadata['sha256']
        assert summary['checkpoint_sha256']==metadata['source_checkpoint_sha256']
        rows.append(dict(product=product,condition=condition,epoch=1350,
            path=str(source.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size,
            full_checkpoint_sha256=metadata['source_checkpoint_sha256'],
            metrics={k:summary[k] for k in ('mean_joint_f1','mean_raw_onset_f1',
                'mean_raw_grid_bonus','mean_tail_10s_recall','completed','total')},
            reason='Slightly higher onset F1 than the selected model; lower primary joint F1. '
                   'Fixed training songs only, not a new-song quality claim.'))
    metadata=dict(schema='audiolab-ar-milestone-supplement-v1',snapshots=rows,
                  base_manifest_sha256=sha(DEST/'manifest.json'))
    write(DEST/'supplementary-manifest.json',metadata)
    archive=WORK/'AL-AR_additional_e1350.zip'
    if not archive.exists():
        assert shutil.disk_usage(WORK).free>RESERVE+sum(r['bytes'] for r in rows)+2*1024**2
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for row in rows:
                source=ROOT/row['path']
                z.write(source,row['path'])
                z.write(source.with_suffix('.json'),str(Path(row['path']).with_suffix('.json')))
            z.write(DEST/'supplementary-manifest.json','supplementary-manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in rows:
            with z.open(row['path']) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    verified=upload(archive,env)
    assets=[a for a in published['assets'] if a['name']!=verified['name']]+[verified]
    checksum=DEST/'SHA256SUMS-additional.txt'
    checksum.write_text(f"{verified['sha256']}  {verified['name']}\n")
    checked=upload(checksum,env)
    assets=[a for a in assets if a['name']!=checked['name']]+[checked]
    final=release(env)
    by_name={a['name']:a for a in final['assets']}
    assert len(by_name)==12 and final['prerelease'] and not final['draft']
    for asset in assets:
        assert by_name[asset['name']]['size']==asset['bytes']
        assert by_name[asset['name']]['digest']=='sha256:'+asset['sha256']
    published.update(assets=assets,supplement='supplementary-manifest.json')
    write(DEST/'published.json',published)
    notes=DEST/'release-notes.md'
    addition='\n额外对照：AL-AR_additional_e1350.zip 保存 4K S1350 与 free-x A1350；两者在固定 15 首训练曲上的起点 F1 略高，但主联合 F1 低于各自首选。附加包的原文件 SHA 位于包内 supplementary-manifest.json，包校验见 SHA256SUMS-additional.txt。\n'
    if 'AL-AR_additional_e1350.zip' not in notes.read_text():
        notes.write_text(notes.read_text()+addition)
    body=WORK/'release-body.json'
    write(body,{'body':notes.read_text()})
    command('gh','api','--method','PATCH',f'repos/{REPO}/releases/{final["id"]}',
            '--input',str(body),env=env)
    body.unlink()
    archive.unlink()
    print(json.dumps(dict(status='verified',assets=len(assets),supplemental_epochs=rows,
        url=final['html_url']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--supplement']:supplement()
    elif not sys.argv[1:]:publish()
    else:raise SystemExit('Usage: archive.py [--supplement]')
