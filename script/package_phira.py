"""Package an existing completed AudioLab job without model inference."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from src.product.artifacts import write_json
from src.product.phira import build_phira_package, check_resources, sha256, PHIRA_VERSION, PHIRA_COMMIT


def package_job(job_directory, output_directory, mode='both'):
    job=Path(job_directory).resolve();output=Path(output_directory).resolve()
    if output==job or job in output.parents:
        raise ValueError('package existing jobs into a separate directory')
    if mode not in ('both','4k','freex'):raise ValueError('invalid mode')
    manifest_path=job/'manifest.json';manifest_hash=sha256(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='completed':raise ValueError('only completed source jobs can be packaged')
    modes=['4k','freex'] if mode=='both' else [mode]
    if any(m not in manifest['routes'] for m in modes):raise ValueError('requested source route is missing')
    def source_file(name):
        path=job/name
        if Path(name).name!=name or '\\' in name or path.resolve().parent!=job:
            raise ValueError('source artifact must be a regular job file')
        if sha256(path)!=manifest['artifacts'][name]['sha256']:
            raise ValueError('source artifact fingerprint differs from completed manifest')
        return path
    audio=source_file(manifest['audio']['file'])
    if sha256(audio)!=manifest['audio']['sha256']:raise ValueError('source audio fingerprint mismatch')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='audiolab-phira-delivery-v1',status='running',packages={},
                source_manifest_sha256=manifest_hash,target_version=PHIRA_VERSION,target_commit=PHIRA_COMMIT,
                model_inference_performed=False,target_player_run='not_executed',touch_playtest='not_executed')
    try:
        check_resources(output,(len(modes)*audio.stat().st_size)+32*1024**2)
        for selected in modes:
            chart=source_file(manifest['routes'][selected]['chart_file'])
            source=dict(kind='model_generated',generation_manifest_sha256=manifest_hash,
                        generation_commit=manifest['implementation']['git_commit'],
                        model_sha256=manifest['model_sha256'][selected],
                        timebase_source=manifest['config']['timing_source'],
                        seed=manifest['config']['seed'],repair_profile='static-v1')
            report['packages'][selected]=build_phira_package(chart,audio,output/f'{selected}.phira.zip',
                mode=selected,title=manifest['title'],duration_seconds=manifest['audio']['duration_seconds'],source=source)
        if sha256(manifest_path)!=manifest_hash:raise ValueError('source manifest changed during packaging')
        links=''.join(f'<p><a download data-mode="{m}" href="{p["file"]}">下载 Phira {html.escape(m)} 谱包</a></p>'
                      for m,p in report['packages'].items())
        (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>AudioLab · Phira 谱包</title>'
            '<style>body{font:18px system-ui;max-width:800px;margin:40px auto;padding:20px;line-height:1.8}a{color:#146eb4}</style>'
            f'<h1>{html.escape(manifest["title"])}</h1>'+links+
            '<p>下载完整 ZIP 并传到手机或平板，在 Phira 本地谱面中导入。4K 为四个固定触屏位置；free-x 为静态自由位置。'
            '两种模式保留四键型与多指。谱包尚未真机验证，难度未定级。</p>'
            '<p><a download href="packages.json">下载来源与包校验报告</a></p></html>',encoding='utf-8')
        report['status']='completed';write_json(output/'packages.json',report)
        if json.loads((output/'packages.json').read_text())!=report:raise ValueError('package report readback differs')
        check_resources(output)
        return report
    except BaseException as error:
        for name in ['packages.json','index.html',*[f'{m}.phira.zip' for m in modes]]:
            (output/name).unlink(missing_ok=True)
        try:write_json(output/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        except OSError:pass
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--job',required=True,help='existing completed generation directory')
    result.add_argument('--output',required=True,help='new separate package directory')
    result.add_argument('--mode',choices=['both','4k','freex'],default='both')
    return result


if __name__=='__main__':
    args=parser().parse_args()
    report=package_job(args.job,args.output,args.mode)
    print(json.dumps(dict(status=report['status'],packages={m:dict(file=p['file'],sha256=p['sha256'],bytes=p['bytes'])
                                                          for m,p in report['packages'].items()})))
