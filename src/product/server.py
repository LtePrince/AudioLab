"""Loopback-only, single-job local interface; workers own inference memory."""
from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from src.product.artifacts import write_json
from .limits import (DEFAULT_MIN_AVAILABLE_RAM_GIB, ram_reserve_argument,
                     validate_ram_reserve)

MAX_UPLOAD = 128*1024**2
MIN_DISK = 8*1024**3
JOB_ID = re.compile(r'^[0-9a-f]{32}$')


class BusyError(ValueError):
    pass


def valid_name(name):
    if not name or Path(name).name != name or '\\' in name or '\x00' in name or len(name.encode()) > 200:
        raise ValueError('invalid audio filename')
    if Path(name).suffix.lower() not in ('.wav','.ogg'):
        raise ValueError('only WAV and OGG are supported')
    return name


def parse_range(header, size):
    if not header:
        return 0,size-1,False
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError('invalid byte range')
    a,b=match.groups()
    if not a:
        length=int(b)
        if length <= 0:raise ValueError('invalid suffix range')
        return max(0,size-length),size-1,True
    start=int(a);end=min(int(b),size-1) if b else size-1
    if start>=size or end<start:raise ValueError('unsatisfiable byte range')
    return start,end,True


class LocalJobs:
    def __init__(self, root, model_root, device, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
        self.min_available_ram_gib=validate_ram_reserve(min_available_ram_gib)
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.model_root=str(Path(model_root).resolve());self.device=device
        for part in ['jobs','uploads','cancel','records','logs']:(self.root/part).mkdir(exist_ok=True)
        self.lock=threading.RLock();self.active=None;self.process=None;self.monitor_thread=None;self.records={}
        self.lease=(self.root/'.server.lock').open('a')
        try:fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another local server owns this output directory')
        for file in (self.root/'records').glob('*.json'):
            record=json.loads(file.read_text());identifier=record['id']
            if not JOB_ID.fullmatch(identifier):continue
            if record['status'] in ('uploading','queued','running','cancelling'):
                pid=record.get('pid');cmdline=Path(f'/proc/{pid}/cmdline') if pid else None
                if cmdline and cmdline.exists() and str(self.root/'jobs'/identifier).encode() in cmdline.read_bytes():
                    raise RuntimeError(f'previous generation process {pid} still owns an unfinished job')
                record.update(status='failed',error='previous server stopped before completion')
                write_json(file,record)
            self.records[identifier]=record

    def save(self,record):
        write_json(self.root/'records'/f"{record['id']}.json",record)

    def reserve(self,name,mode,bpm,size,request_token=None):
        name=valid_name(name)
        if mode not in ('4k','freex','both'):raise ValueError('invalid mode')
        if bpm is not None and (not isinstance(bpm,(float,int)) or not 1<=bpm<=1000):
            raise ValueError('BPM must be finite in [1,1000]')
        if not 0<size<=MAX_UPLOAD:raise ValueError('audio must be between 1 byte and 128 MiB')
        if request_token is not None and not re.fullmatch(r'[a-zA-Z0-9-]{1,64}',request_token):
            raise ValueError('invalid request token')
        with self.lock:
            if self.active is not None:raise BusyError('a generation job is still active')
            # Upload + immutable audio snapshot + one complete package per mode.
            copies=4 if mode=='both' else 3
            if shutil.disk_usage(self.root).free < MIN_DISK+copies*size+64*1024**2:
                raise ValueError('insufficient disk reserve for upload and generation')
            identifier=uuid.uuid4().hex
            record=dict(id=identifier,name=name,mode=mode,bpm=bpm,status='uploading',created=time.time(),request_token=request_token,
                        resource_limits=dict(min_available_ram_gib=self.min_available_ram_gib))
            folder=self.root/'uploads'/identifier;folder.mkdir()
            self.save(record)
            self.records[identifier]=record;self.active=identifier
            return identifier,folder/name

    def upload_check(self,identifier):
        if (self.root/'cancel'/identifier).exists():raise InterruptedError('upload cancelled')
        if shutil.disk_usage(self.root).free < MIN_DISK:raise ValueError('disk reserve reached')

    def fail(self,identifier,error,status='failed'):
        with self.lock:
            if self.active==identifier and self.process is not None and self.process.poll() is None:
                # Only monitor(), after wait(), may release a live worker slot.
                (self.root/'cancel'/identifier).touch()
                record=self.records[identifier];record.update(status='cancelling',error=str(error));self.save(record)
                return
            record=self.records[identifier];record.update(status=status,error=str(error));self.save(record)
            if self.active==identifier:self.active=None
        self.cleanup_upload(identifier)

    def cleanup_upload(self,identifier):
        folder=self.root/'uploads'/identifier
        if folder.exists():
            for file in folder.iterdir():
                if file.is_file():file.unlink()
            folder.rmdir()

    def start(self,identifier,audio):
        with self.lock:
            record=self.records[identifier]
            self.upload_check(identifier)
            output=self.root/'jobs'/identifier
            command=[sys.executable,'-m','script.generate_product','--audio',str(audio),
                     '--mode',record['mode'],'--model-root',self.model_root,'--output',str(output),
                     '--device',self.device,'--cancel-file',str(self.root/'cancel'/identifier),
                     '--min-available-ram-gib',str(self.min_available_ram_gib)]
            if record['bpm'] is not None:command+=['--bpm',str(record['bpm'])]
            with (self.root/'logs'/f'{identifier}.log').open('wb') as log:
                process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=log)
            self.process=process;record.update(status='running',pid=process.pid)
            try:
                self.save(record)
                self.monitor_thread=threading.Thread(target=self.monitor,args=(identifier,process),daemon=True)
                self.monitor_thread.start()
            except (OSError,RuntimeError):
                process.terminate();process.wait(timeout=15);self.process=None;raise

    def monitor(self,identifier,process):
        code=process.wait()
        folder=self.root/'jobs'/identifier
        try:
            state=json.loads((folder/'job.json').read_text())
        except (OSError,ValueError):state=dict(status='failed',error=dict(message=f'worker exited with code {code}'))
        completed=(folder/'manifest.json').exists() and state.get('status')=='completed' and code==0
        status='completed' if completed else state.get('status','failed')
        if status not in ('cancelled','failed','needs_tempo','completed'):status='failed'
        if status=='completed' and not completed:status='failed'
        with self.lock:
            record=self.records[identifier];record.update(status=status,exit_code=code,finished=time.time())
            if not completed:record['error']=state.get('error',dict(message='worker did not complete'))
            try:
                self.save(record)
                self.cleanup_upload(identifier)
            finally:
                if self.active==identifier:self.active=None;self.process=None

    def status(self,identifier):
        with self.lock:
            if identifier not in self.records:raise KeyError(identifier)
            record=dict(self.records[identifier]);record['busy']=self.active is not None
        path=self.root/'jobs'/identifier/'job.json'
        if path.exists():
            state=json.loads(path.read_text())
            record['progress']=state.get('progress');record['stage']=state.get('stage')
            record['elapsed_seconds']=state.get('elapsed_seconds')
            # The process must have exited before the slot/finished UI is released.
            if record['status'] not in ('completed','failed','cancelled','needs_tempo'):
                record['worker_status']=state.get('status')
        if record['status']=='completed':
            record['preview']=f'/jobs/{identifier}/preview.html'
            record['report']=f'/jobs/{identifier}/manifest.json'
            manifest=json.loads((self.root/'jobs'/identifier/'manifest.json').read_text())
            record['warnings']=manifest.get('warnings', [])
            record['packages']={mode:dict(url=f'/jobs/{identifier}/{package["file"]}',
                                         sha256=package['sha256'],bytes=package['bytes'])
                                for mode,package in manifest.get('packages',{}).items()}
        return record

    def cancel(self,identifier):
        with self.lock:
            if self.active!=identifier:raise ValueError('job is no longer active')
            (self.root/'cancel'/identifier).touch()
            self.records[identifier]['status']='cancelling';self.save(self.records[identifier])

    def artifact(self,identifier,name):
        if not JOB_ID.fullmatch(identifier) or Path(name).name!=name:raise FileNotFoundError()
        state=self.status(identifier)
        if state['status']!='completed':raise FileNotFoundError()
        folder=self.root/'jobs'/identifier
        manifest=json.loads((folder/'manifest.json').read_text())
        if name!='manifest.json' and name not in manifest['artifacts']:raise FileNotFoundError()
        path=folder/name
        if path.resolve().parent!=folder.resolve():raise FileNotFoundError()
        return path

    def close(self):
        with self.lock:
            identifier,process=self.active,self.process
            if identifier:(self.root/'cancel'/identifier).touch()
        if process:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=15)
        if self.monitor_thread:self.monitor_thread.join(timeout=5)
        self.lease.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    @property
    def jobs(self):return self.server.jobs

    def local_request(self):
        expected={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in expected:return False
        origin=self.headers.get('Origin')
        return origin is None or origin in {'http://'+host for host in expected}

    def reply(self,status,value):
        body=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('Connection','close');self.end_headers();self.wfile.write(body);self.close_connection=True

    def send_file(self,path):
        size=path.stat().st_size
        try:start,end,partial=parse_range(self.headers.get('Range'),size)
        except ValueError:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
            self.send_header('Content-Length','0');self.end_headers();return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        self.send_header('X-Content-Type-Options','nosniff')
        if path.suffix=='.zip':self.send_header('Content-Disposition',f'attachment; filename="{path.name}"')
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start);remaining=end-start+1
            while remaining:
                block=stream.read(min(64*1024,remaining))
                if not block:break
                self.wfile.write(block);remaining-=len(block)

    def do_GET(self):
        if not self.local_request():self.reply(403,dict(error='local origin required'));return
        path=unquote(urlsplit(self.path).path)
        try:
            if path=='/':self.send_file(Path(__file__).with_name('app.html'))
            elif path=='/api/jobs':
                with self.jobs.lock:ids=sorted(self.jobs.records,key=lambda i:self.jobs.records[i]['created'],reverse=True)[:20]
                self.reply(200,dict(jobs=[self.jobs.status(i) for i in ids]))
            elif path.startswith('/api/jobs/'):
                identifier=path.removeprefix('/api/jobs/');self.reply(200,self.jobs.status(identifier))
            elif path.startswith('/jobs/'):
                _,_,identifier,name=path.split('/',3);self.send_file(self.jobs.artifact(identifier,name))
            else:self.reply(404,dict(error='not found'))
        except (ValueError,KeyError,OSError):self.reply(404,dict(error='not found'))

    def do_POST(self):
        if not self.local_request():self.reply(403,dict(error='local origin required'));return
        parsed=urlsplit(self.path);path=unquote(parsed.path)
        if path.startswith('/api/jobs/') and path.endswith('/cancel'):
            identifier=path[len('/api/jobs/'):-len('/cancel')]
            try:self.jobs.cancel(identifier);self.reply(202,dict(status='cancelling'))
            except ValueError as error:self.reply(409,dict(error=str(error)))
            return
        if path!='/api/jobs':self.reply(404,dict(error='not found'));return
        identifier=None
        try:
            if self.headers.get('Transfer-Encoding'):raise ValueError('streaming transfer encoding is not supported')
            size=int(self.headers.get('Content-Length','0'))
            query=parse_qs(parsed.query)
            name=query.get('name',[''])[0];mode=query.get('mode',['both'])[0]
            raw_bpm=query.get('bpm',['auto'])[0]
            bpm=None if raw_bpm=='auto' else float(raw_bpm)
            identifier,target=self.jobs.reserve(name,mode,bpm,size,query.get('request_token',[None])[0])
            self.connection.settimeout(30)
            with target.open('xb') as stream:
                remaining=size
                while remaining:
                    self.jobs.upload_check(identifier)
                    chunk=self.rfile.read(min(64*1024,remaining))
                    if not chunk:raise ValueError('incomplete upload')
                    stream.write(chunk);remaining-=len(chunk)
            self.jobs.start(identifier,target)
        except BusyError as error:self.reply(409,dict(error=str(error)))
        except (ValueError,OSError,RuntimeError) as error:
            if identifier:self.jobs.fail(identifier,error,'cancelled' if isinstance(error,InterruptedError) else 'failed')
            self.reply(400,dict(error=str(error),id=identifier))
        else:
            # The worker now owns the slot. A disconnected upload client must
            # not release it or delete its input while inference is running.
            try:self.reply(202,dict(id=identifier,status='running'))
            except OSError:self.close_connection=True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',required=True);p.add_argument('--output',default='out/local-product')
    p.add_argument('--device',default='cpu');p.add_argument('--port',type=int,default=7860)
    p.add_argument('--min-available-ram-gib',type=ram_reserve_argument,default=DEFAULT_MIN_AVAILABLE_RAM_GIB)
    return p


def main():
    args=parser().parse_args()
    jobs=LocalJobs(args.output,args.model_root,args.device,args.min_available_ram_gib)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.jobs=jobs
    def stop(_signal,_frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    print(f'AudioLab local interface: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();jobs.close()


if __name__=='__main__':main()
