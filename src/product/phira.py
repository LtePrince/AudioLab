"""Bounded, self-contained Phira v0.8.2 packages for validated static charts.

Compatibility checks follow pinned source contracts; they do not run Phira or
certify touch playability. Chart/audio bytes and generation policy are unchanged.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import zipfile
import zlib

import soundfile as sf

from src.audio_understanding import SAMPLE_RATE
from .artifacts import validate_chart
from .limits import DEFAULT_MIN_AVAILABLE_RAM_GIB, validate_ram_reserve

PHIRA_VERSION = '0.8.2'
PHIRA_COMMIT = 'c830dee5e07a2b8f06104576f30d8045e0859aa7'
SCHEMA = 'audiolab-phira-package-v1'
MIB = 1024**2
GIB = 1024**3
MAX_AUDIO = 128*MIB
MAX_CHART = 8*MIB
MAX_PACKAGE = 160*MIB
MODES = {'4k': '4K', 'freex': 'free-x'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def resource_state(directory):
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    rss = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    return dict(disk_free_gib=shutil.disk_usage(directory).free/GIB,
                ram_available_gib=memory['MemAvailable']/GIB,rss_gib=rss/GIB)


def check_resources(directory, extra_bytes=0, *, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    min_available_ram_gib=validate_ram_reserve(min_available_ram_gib)
    state=resource_state(directory)
    if state['disk_free_gib'] < 8+extra_bytes/GIB:
        raise RuntimeError(f'Resource budget: insufficient disk reserve for Phira package: {state}')
    if state['ram_available_gib'] < min_available_ram_gib:
        raise RuntimeError('Resource budget: insufficient available RAM for Phira package: '
                           f'min_available_ram_gib={min_available_ram_gib:g}, {state}')
    if state['rss_gib'] > 4:
        raise RuntimeError(f'Resource budget: process RSS limit exceeded for Phira package: {state}')


@lru_cache(maxsize=1)
def default_cover():
    """Small original neutral cover, generated from a fixed geometric design."""
    width, height = 480, 270
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            color = (18+y//30, 29+y//18, 48+y//12)
            if 204 <= y <= 206 and 36 <= x < 444:
                color = (91, 119, 156)
            for anchor, shade in zip((138,206,274,342), ((140,216,255),(255,217,124),(190,156,255),(255,146,184))):
                if abs(x-anchor)<12 and 197<=y<=202:color=shade
            rows.extend(color)
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(bytes(rows),6))+chunk(b'IEND',b''))


def chart_contract(chart_bytes, mode, duration):
    if len(chart_bytes)>MAX_CHART or not chart_bytes.startswith(b'{'):
        raise ValueError('Phira PGR requires a bounded UTF-8 JSON object without BOM')
    chart=json.loads(chart_bytes)
    # GameScene infers RPE from this literal anywhere in the source text.
    if b'"META"' in chart_bytes:raise ValueError('RPE META marker is outside this static PGR profile')
    line=chart['judgeLineList'][0]
    notes=line['notesAbove']+line['notesBelow']
    events,checks=validate_chart(chart,mode,duration,{n['time'] for n in notes})
    # Required serde fields and the exact static geometry used by the product.
    for key,expected in [('judgeLineMoveEvents',dict(start=.5,end=.5,start2=.1,end2=.1)),
                         ('judgeLineRotateEvents',dict(start=0.,end=0.)),
                         ('judgeLineDisappearEvents',dict(start=1.,end=1.))]:
        segments=line[key]
        if (len(segments)!=2 or segments[0]['startTime']!=-999999 or segments[0]['endTime']!=0
                or segments[1]['startTime']!=0 or segments[1]['endTime']!=1000000000):
            raise ValueError('package requires the generated static line event profile')
        for segment in segments:
            if (not all(math.isfinite(segment[k]) for k in ('startTime','endTime',*expected))
                    or segment['startTime']>segment['endTime']
                    or any(segment[k]!=v for k,v in expected.items())):
                raise ValueError('package requires the validated static line profile')
    speeds=line['speedEvents']
    if not speeds or any(not all(math.isfinite(e[k]) for k in ('startTime','endTime','value'))
                         or e['startTime']>e['endTime'] for e in speeds):
        raise ValueError('invalid speed events for Phira')
    if (speeds[0]['startTime']!=0 or speeds[0]['value']<=0
            or speeds[0]['endTime']<max(n['time']+n['holdTime'] for n in notes)):
        raise ValueError('static speed segment does not cover every note and Hold tail')
    if (len(speeds)!=2 or speeds[1]['startTime']!=speeds[0]['endTime']
            or speeds[1]['endTime']!=1000000000 or speeds[1]['value']!=0):
        raise ValueError('static speed profile requires one positive segment and its zero-speed sentinel')
    return chart,dict(**checks,format='pgr',format_version=3,
                      chart_offset_seconds=chart['offset'],metadata_offset_seconds=0.,
                      bpm=line['bpm'],
                      verification='static PGR/source loading conditions; not Phira execution'),events


def audio_contract(path, duration, check, *, audio_name=None):
    if not math.isfinite(duration) or not 0<duration<=600:
        raise ValueError('package audio duration must be within 600 seconds')
    with sf.SoundFile(path) as audio:
        if not 1<=audio.channels<=8 or not 1<=audio.samplerate<=192000:
            raise ValueError('audio channel count or sample rate exceeds the bounded decoder profile')
        actual=audio.frames/audio.samplerate
        # librosa's 22050-Hz analysis waveform has ceil-resampled length. Keep
        # the original soundtrack bytes and admit at most one analysis sample.
        tolerance=1/SAMPLE_RATE+1e-10
        if abs(actual-duration)>tolerance or actual<=0:
            raise ValueError('package audio duration does not match the chart job')
        expected='OGG' if Path(audio_name or path).suffix.lower()=='.ogg' else 'WAV'
        if audio.format!=expected:
            raise ValueError('audio container differs from WAV/OGG filename')
        frames=0
        while len(block:=audio.read(65536,dtype='float32',always_2d=True)):
            check();frames+=len(block)
            # No waveform copy/transcode; reject malformed/non-finite samples.
            import numpy as np
            if not np.isfinite(block).all():raise ValueError('non-finite decoded audio samples')
        if frames!=audio.frames:raise ValueError('audio decoder did not reach the declared end')
        return dict(duration_seconds=actual,analysis_duration_seconds=duration,
                    analysis_minus_source_seconds=duration-actual,duration_tolerance_seconds=tolerance,
                    sample_rate=audio.samplerate,channels=audio.channels,
                    container=audio.format,subtype=audio.subtype,
                    verification='libsndfile full streamed decode; Phira decoder not executed')


def metadata(title, mode, audio_name, duration, source_kind='model_generated'):
    if not isinstance(title,str) or not title.strip() or len(title)>240 or any(ord(c)<32 for c in title):
        raise ValueError('invalid package title')
    if source_kind not in ('model_generated','synthetic_calibration'):
        raise ValueError('unsupported package source kind')
    synthetic=source_kind=='synthetic_calibration'
    return dict(name=f'{title} · {MODES[mode]}',difficulty=0.,level=f'未定级 · {MODES[mode]}',
                charter='AudioLab · 合成校准' if synthetic else 'AudioLab · 模型生成',
                composer='未提供',illustrator='AudioLab · 默认封面',
                chart='chart.json',music=audio_name,illustration='illustration.png',
                previewStart=0.,previewEnd=min(15.,duration),aspectRatio=16/9,backgroundDim=.6,
                lineLength=6.,offset=0.,holdPartialCover=False,
                tip='合成校准谱：检查导入与设备同步，不代表模型效果。' if synthetic else
                    '基线生成谱，请在手机或平板上检查同步与手感。',
                tags=['AudioLab','unrated',mode],
                intro=('人工构造的合成校准谱，不是模型生成或质量评测样本。' if synthetic else '未定级的模型生成候选。')+
                      '手机/平板触屏；保留四种键型与多指。实际手感待确认。')


def member_info(name):
    info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3
    info.external_attr=(stat.S_IFREG|0o644)<<16
    return info


def verify_package(path, *, check=lambda: None):
    """Read all ZIP members, CRCs and fingerprints without extracting files.

    Only our bounded profile is accepted. This is a source-contract check, not
    a universal Phira validator or a device test.
    """
    path=Path(path)
    if path.stat().st_size>MAX_PACKAGE:raise ValueError('Phira package exceeds 160 MiB')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        if len(entries)!=5 or len(set(names))!=5:
            raise ValueError('package must contain five unique root files')
        if any(Path(name).name!=name or '\\' in name or name in ('.','..') for name in names):
            raise ValueError('unsafe package member path')
        if set(names)-{'info.yml','chart.json','illustration.png','audiolab.json','audio.wav','audio.ogg'}:
            raise ValueError('unexpected package member')
        if not {'info.yml','chart.json','illustration.png','audiolab.json'}<=set(names):
            raise ValueError('missing package member')
        limits={'chart.json':MAX_CHART,'info.yml':65536,'audiolab.json':128*1024,'illustration.png':MIB}
        if (sum(entry.file_size for entry in entries)>MAX_PACKAGE
                or any(entry.file_size>limits.get(entry.filename,MAX_AUDIO) or entry.file_size<=0
                       or entry.is_dir() or entry.flag_bits&1
                       or stat.S_ISLNK(entry.external_attr>>16)
                       or entry.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for entry in entries)):
            raise ValueError('unsupported member type or decompressed size')
        small={};fingerprints={}
        for entry in entries:
            digest=hashlib.sha256();count=0;parts=[]
            with archive.open(entry) as stream:
                while block:=stream.read(MIB):
                    check();digest.update(block);count+=len(block)
                    if entry.filename in limits:parts.append(block)
            if count!=entry.file_size:raise ValueError('incomplete ZIP member')
            fingerprints[entry.filename]=dict(sha256=digest.hexdigest(),bytes=count)
            if parts:small[entry.filename]=b''.join(parts)
        # JSON flow syntax is valid YAML and is accepted by serde_yaml::from_str.
        info=json.loads(small['info.yml']);manifest=json.loads(small['audiolab.json'])
        if (manifest['schema']!=SCHEMA or manifest['target']['commit']!=PHIRA_COMMIT
                or manifest['mode'] not in MODES):raise ValueError('unsupported package manifest')
        expected=metadata(manifest['title'],manifest['mode'],manifest['audio_file'],manifest['duration_seconds'],
                          manifest['source_kind'])
        if manifest['source_kind']!=manifest['source'].get('kind','model_generated'):
            raise ValueError('package source kind differs from provenance')
        if info!=expected:raise ValueError('metadata is inconsistent with package provenance')
        for key in ('chart','music','illustration'):
            if info[key] not in names:raise ValueError('metadata references a missing file')
        if manifest['members']!={name:value for name,value in fingerprints.items() if name!='audiolab.json'}:
            raise ValueError('package member fingerprint mismatch')
        chart,contract,_=chart_contract(small['chart.json'],manifest['mode'],manifest['duration_seconds'])
        if contract!=manifest['chart_checks']:raise ValueError('chart contract changed on readback')
        if small['illustration.png']!=default_cover():raise ValueError('default cover bytes differ')
        with archive.open(info['music']) as audio_stream:
            actual_audio=audio_contract(audio_stream,manifest['duration_seconds'],check,audio_name=info['music'])
        if actual_audio!=manifest['audio_checks']:raise ValueError('audio contract changed on readback')
        return dict(status='passed',members=fingerprints,chart_checks=contract,
                    audio_checks=manifest['audio_checks'],difficulty_semantics='unrated_placeholder',
                    target_player_run='not_executed',touch_playtest='not_executed',
                    chart_offset_seconds=chart['offset'],metadata_offset_seconds=info['offset'])


def build_phira_package(chart_path, audio_path, output_path, *, mode, title,
                        duration_seconds, source=None, check=None,
                        min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    """Create a new package atomically. `check` is an optional cancellation hook.

    Audio and chart are read-only, streamed and fingerprinted before/after.
    `source` should contain portable generation hashes/config, never local paths.
    """
    min_available_ram_gib=validate_ram_reserve(min_available_ram_gib)
    chart_path,audio_path,output=map(Path,(chart_path,audio_path,output_path))
    if mode not in MODES:raise ValueError('invalid package mode')
    if output.exists():raise FileExistsError(output)
    if not output.parent.is_dir():raise ValueError('package output directory must exist')
    if (not 0<chart_path.stat().st_size<=MAX_CHART or not 0<audio_path.stat().st_size<=MAX_AUDIO
            or audio_path.suffix.lower() not in ('.wav','.ogg')):
        raise ValueError('package requires bounded WAV/OGG audio and static chart JSON')
    additional=audio_path.stat().st_size+chart_path.stat().st_size+2*MIB
    check_resources(output.parent,additional,min_available_ram_gib=min_available_ram_gib)
    def progress():
        check_resources(output.parent,min_available_ram_gib=min_available_ram_gib)
        if check:check()
    progress()
    inputs={name:dict(path=path,sha256=sha256(path),bytes=path.stat().st_size)
            for name,path in [('chart.json',chart_path),('audio'+audio_path.suffix.lower(),audio_path)]}
    chart,contract,_=chart_contract(chart_path.read_bytes(),mode,duration_seconds)
    audio_checks=audio_contract(audio_path,duration_seconds,progress)
    audio_name='audio'+audio_path.suffix.lower()
    source_kind=(source or {}).get('kind','model_generated')
    blobs={'info.yml':encoded(metadata(title,mode,audio_name,duration_seconds,source_kind)),
           'illustration.png':default_cover()}
    members={name:{key:value for key,value in data.items() if key!='path'} for name,data in inputs.items()}
    members.update({name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for name,data in blobs.items()})
    portable=dict(schema=SCHEMA,mode=mode,title=title,duration_seconds=duration_seconds,audio_file=audio_name,
                  target=dict(name='Phira',version=PHIRA_VERSION,commit=PHIRA_COMMIT),
                  difficulty_semantics='unrated_placeholder',source_kind=source_kind,source=source or {},members=members,
                  chart_checks=contract,audio_checks=audio_checks,
                  timing='audio_seconds = native_tick * 60 / (32 * line_bpm) + chart.offset; info.offset=0',
                  touch_playtest='not_executed',target_player_run='not_executed')
    blobs['audiolab.json']=encoded(portable)
    temporary=output.with_suffix(output.suffix+'.tmp')
    created=False;installed=False
    try:
        with temporary.open('xb') as handle:
            created=True
            with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
                for name,data in blobs.items():archive.writestr(member_info(name),data)
                for name,data in inputs.items():
                    copied=hashlib.sha256()
                    with data['path'].open('rb') as incoming,archive.open(member_info(name),'w') as outgoing:
                        while block:=incoming.read(MIB):
                            progress();copied.update(block);outgoing.write(block)
                    if copied.hexdigest()!=data['sha256']:raise ValueError('package source changed during streaming')
            handle.flush();os.fsync(handle.fileno())
        verified=verify_package(temporary,check=progress)
        if any(sha256(data['path'])!=data['sha256'] for data in inputs.values()):
            raise ValueError('package source changed during verification')
        progress()
        # Hard-link installation is atomic and refuses an existing destination.
        os.link(temporary,output);installed=True
        return dict(file=output.name,sha256=sha256(output),bytes=output.stat().st_size,
                    target_version=PHIRA_VERSION,target_commit=PHIRA_COMMIT,verification=verified,
                    resource_limits=dict(min_available_ram_gib=min_available_ram_gib))
    except BaseException:
        if installed:output.unlink(missing_ok=True)
        raise
    finally:
        if created:temporary.unlink(missing_ok=True)
