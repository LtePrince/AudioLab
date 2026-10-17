"""Mock measurements, real RAM guards and bounded CPU model/package execution."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import resource_budget
from src.audio_understanding import inference
from src.product import runtime, server, pipeline, phira
from src.product.limits import validate_ram_reserve
from script import package_phira, generate_product
from test.test_product_pipeline import inputs, arguments


@pytest.fixture
def pc_memory(monkeypatch):
    # The reported user-PC measurements, without consuming RAM to reproduce them.
    state = dict(disk_free_gib=116.9947, ram_available_gib=7.892978668, rss_gib=1.2777138)
    for module in (resource_budget, pipeline, inference):
        monkeypatch.setattr(module, 'resource_snapshot', lambda path: dict(state))
    for module in (runtime, phira):
        monkeypatch.setattr(module, 'resource_state', lambda path: dict(state))
    return state


@pytest.mark.parametrize('invalid', [0, -.5, .999, float('nan'), float('inf'), -float('inf'), True])
def test_invalid_ram_settings_cannot_disable_protection(invalid):
    with pytest.raises(ValueError, match='finite and at least 1'):
        validate_ram_reserve(invalid)


def test_explicit_valid_reserve_keeps_default_and_other_resource_limits(tmp_path, pc_memory):
    assert [validate_ram_reserve(v) for v in (1,4,'8')] == [1.,4.,8.]
    with pytest.raises(runtime.PreflightError, match='内存'):
        runtime.check_resources(tmp_path)
    with pytest.raises(RuntimeError, match='insufficient available RAM'):
        phira.check_resources(tmp_path)
    runtime.check_resources(tmp_path,4)
    phira.check_resources(tmp_path,min_available_ram_gib=4)
    pc_memory['ram_available_gib']=3.9
    with pytest.raises(runtime.PreflightError):runtime.check_resources(tmp_path,4)
    with pytest.raises(RuntimeError,match='min_available_ram_gib=4'):
        phira.check_resources(tmp_path,min_available_ram_gib=4)
    pc_memory['ram_available_gib']=7.892978668
    for field,value,phrase in [('disk_free_gib',7.99,'disk reserve'),('rss_gib',4.01,'process RSS limit exceeded')]:
        previous=pc_memory[field];pc_memory[field]=value
        with pytest.raises(runtime.PreflightError):runtime.check_resources(tmp_path,4)
        with pytest.raises(RuntimeError,match=phrase):phira.check_resources(tmp_path,min_available_ram_gib=4)
        pc_memory[field]=previous


def test_runtime_check_and_launch_forward_same_reserve(tmp_path, monkeypatch):
    observed=[]
    monkeypatch.setattr(runtime,'preflight',lambda *args:(observed.append(args[-1]) or
                        dict(status='ready',checks=dict(device=dict(resolved='cuda:0')))))
    monkeypatch.setattr(runtime.os,'chdir',lambda path:None)
    monkeypatch.setattr(runtime.os,'execv',lambda executable, command:observed.append(command))
    assert runtime.main(['--check','--output',str(tmp_path),'--min-available-ram-gib','4'])==0
    assert runtime.main(['--output',str(tmp_path),'--min-available-ram-gib','4'])==0
    assert observed[:2]==[4.,4.]
    command=observed[-1]
    assert float(command[command.index('--min-available-ram-gib')+1])==4
    assert command[command.index('--device')+1]=='cuda:0'
    default=runtime.server_command(tmp_path/'models',tmp_path/'out','cpu',7860)
    assert float(default[default.index('--min-available-ram-gib')+1])==8


@pytest.mark.parametrize('reserve',[4,4.123456789])
def test_server_passes_reserve_to_real_worker_argument_parser(tmp_path, monkeypatch, reserve):
    commands=[]
    process=SimpleNamespace(pid=12345,poll=lambda:0,wait=lambda **kwargs:0)
    monkeypatch.setattr(server.subprocess,'Popen',lambda command,**kwargs:(commands.append(command) or process))
    monkeypatch.setattr(server.threading,'Thread',lambda **kwargs:Mock())
    monkeypatch.setattr(server.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*1024**3))
    launch=runtime.server_command(tmp_path/'models',tmp_path/'out','cuda:0',7860,reserve)
    settings=server.parser().parse_args(launch[3:])
    assert settings.min_available_ram_gib==reserve
    jobs=server.LocalJobs(tmp_path/'out',tmp_path/'models','cuda:0',settings.min_available_ram_gib)
    try:
        identifier,audio=jobs.reserve('song.wav','both',120,100)
        jobs.start(identifier,audio)
        args=generate_product.parser().parse_args(commands[0][3:])
        assert args.min_available_ram_gib==reserve
        assert args.max_rss_gib==4 and args.min_free_disk_gib==8 and args.gpu_memory_fraction==.65
        assert jobs.records[identifier]['resource_limits']['min_available_ram_gib']==reserve
    finally:jobs.close()
    assert server.parser().parse_args(['--model-root','models']).min_available_ram_gib==8
    assert package_phira.parser().parse_args(['--job','job','--output','out']).min_available_ram_gib==8


@pytest.mark.parametrize('invalid',['0','nan','inf'])
def test_entry_cli_rejects_invalid_reserve_before_start(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(runtime,'preflight',lambda *a:pytest.fail('invalid parameter reached preflight'))
    with pytest.raises(SystemExit) as error:
        runtime.main(['--check','--min-available-ram-gib',invalid])
    assert error.value.code==2
    with pytest.raises(SystemExit):server.parser().parse_args(['--model-root','models','--min-available-ram-gib',invalid])
    with pytest.raises(SystemExit):package_phira.parser().parse_args(['--job','job','--output','out','--min-available-ram-gib',invalid])
    monkeypatch.setattr(generate_product,'run_job',lambda args:pytest.fail('invalid parameter reached generation'))
    monkeypatch.setattr(sys,'argv',[
        'generate_product','--audio','song.wav','--output',str(tmp_path/'bad'),'--min-available-ram-gib',invalid])
    with pytest.raises(SystemExit):generate_product.main()
    assert not (tmp_path/'bad').exists()


def test_user_ram_with_explicit_four_completes_both_packages_and_readback(tmp_path, inputs, pc_memory, monkeypatch):
    # Keep ResourceBudget, Phira guards, decoding and ZIP readback real. Only
    # resource measurements and tiny deterministic test weights are synthetic.
    seen=[];reading=[]
    original_check=phira.check_resources
    original_verify=phira.verify_package
    def guard(path,extra_bytes=0,**kwargs):
        seen.append((kwargs.get('min_available_ram_gib'),tuple(reading)))
        return original_check(path,extra_bytes,**kwargs)
    def verify(path,**kwargs):
        reading.append(path.name)
        try:return original_verify(path,**kwargs)
        finally:reading.pop()
    monkeypatch.setattr(phira,'check_resources',guard)
    monkeypatch.setattr(phira,'verify_package',verify)
    args=arguments(tmp_path,inputs,min_available_ram_gib=4,min_free_disk_gib=8,max_rss_gib=4)
    report=pipeline.run_job(args)
    folder=tmp_path/'job'
    assert report['status']=='completed' and set(report['packages'])=={'4k','freex'}
    assert report['config']['resource_limits']==dict(
        min_available_ram_gib=4,min_free_disk_gib=8,max_rss_gib=4,gpu_memory_fraction=.65)
    assert json.loads((folder/'job.json').read_text())['resource_limits']['min_available_ram_gib']==4
    assert all(limit==4 for limit,_ in seen)
    for mode in ('4k','freex'):
        assert any(f'{mode}.phira.zip.tmp' in stages for _,stages in seen)
        assert report['packages'][mode]['resource_limits']['min_available_ram_gib']==4
        assert original_verify(folder/f'{mode}.phira.zip')['status']=='passed'
    # The standalone packaging entry has its own start/end checks as well.
    standalone=package_phira.package_job(folder,tmp_path/'repack',min_available_ram_gib=4)
    assert standalone['status']=='completed' and len(standalone['packages'])==2
    assert standalone['resource_limits']['min_available_ram_gib']==4
    with pytest.raises(RuntimeError,match='insufficient available RAM'):
        package_phira.package_job(folder,tmp_path/'default-repack')
    assert not list((tmp_path/'default-repack').glob('*.zip'))
    assert not (tmp_path/'default-repack/packages.json').exists()


def test_same_user_ram_default_eight_fails_with_no_success_artifact(tmp_path, inputs, pc_memory):
    args=arguments(tmp_path,inputs,min_available_ram_gib=8,min_free_disk_gib=8,max_rss_gib=4)
    with pytest.raises(RuntimeError,match='insufficient available RAM'):
        pipeline.run_job(args)
    folder=tmp_path/'job'
    state=json.loads((folder/'job.json').read_text())
    assert state['status']=='failed' and state['resource_limits']['min_available_ram_gib']==8
    assert not (folder/'manifest.json').exists() and not list(folder.glob('*.zip'))
