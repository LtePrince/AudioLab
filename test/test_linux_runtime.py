"""Local runtime contracts; no installation, real resource exhaustion or GPU."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.product import runtime
from src.product import model_bundle
from test.test_product_pipeline import inputs


@pytest.fixture(autouse=True)
def controlled_resources(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(runtime, 'resource_state', lambda path: dict(
        disk_free_gib=20, ram_available_gib=20, rss_gib=.1))
    monkeypatch.setattr(model_bundle, '_check_space', lambda *a, **k: None)


def test_bootstrap_help_needs_only_standard_library():
    process = subprocess.run([sys.executable, '-S', '-m', 'script.run_product', '--help'],
                             cwd=runtime.REPO_ROOT, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert '--model-root' in process.stdout and '--check' in process.stdout
    process = subprocess.run([sys.executable, '-S', '-c',
        'from src.product import runtime; import sys; assert "torch" not in sys.modules; '
        'runtime.dependency_versions(runtime.REPO_ROOT/"requirements/runtime.txt")'],
        cwd=runtime.REPO_ROOT, text=True, capture_output=True)
    assert process.returncode != 0 and '缺少运行依赖' in process.stderr


def test_missing_dependencies_stop_before_imports_and_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'basic_environment', lambda *a: None)
    def missing(_name):
        raise importlib.metadata.PackageNotFoundError()
    monkeypatch.setattr(runtime.importlib.metadata, 'version', missing)
    monkeypatch.setattr(runtime, 'model_files', lambda *a: pytest.fail('model access before dependencies'))
    report = runtime.preflight(tmp_path/'models', tmp_path, 'cuda', 7860)
    assert report['status'] == 'failed'
    assert report['error']['code'] == 'dependencies_missing'
    assert 'requirements/runtime.txt' in report['error']['action']


def test_torch_build_suffix_allowed_but_mixed_family_rejected(tmp_path, monkeypatch):
    requirements = tmp_path/'requirements.txt'
    requirements.write_text('torch==2.10.0\ntorchaudio==2.10.0\ntorchcodec==0.10.0\n')
    versions = dict(torch='2.10.0+cu126', torchaudio='2.10.0+cu126', torchcodec='0.10.0')
    monkeypatch.setattr(runtime.importlib.metadata, 'version', versions.__getitem__)
    assert all(value['matches_baseline'] for value in runtime.dependency_versions(requirements).values())
    versions['torchaudio'] = '2.9.0'
    with pytest.raises(runtime.PreflightError, match='组件版本不匹配'):
        runtime.dependency_versions(requirements)


@pytest.mark.parametrize('field,value,code', [
    ('disk_free_gib',8.01,'disk_reserve'), ('ram_available_gib',7.9,'ram_reserve'),
    ('rss_gib',4.1,'ram_reserve')])
def test_resources_refuse_before_large_work(tmp_path, monkeypatch, field, value, code):
    state = dict(disk_free_gib=20,ram_available_gib=20,rss_gib=.1)
    state[field] = value
    monkeypatch.setattr(runtime, 'resource_state', lambda p: state)
    with pytest.raises(runtime.PreflightError) as error:
        runtime.check_resources(tmp_path)
    assert error.value.code == code


def test_missing_weight_is_actionable_before_deserialization(tmp_path):
    with pytest.raises(runtime.PreflightError) as error:
        runtime.model_files(tmp_path)
    assert error.value.code == 'model_missing'
    assert 'model_bundle install' in error.value.action


def test_manifest_reuses_bounded_root_validator_and_detects_tamper(inputs):
    _, root = inputs
    report = model_bundle.export_bundle(root, root.parent/'models.zip', bundle_id='test-runtime')
    # export leaves the source untouched; copy just the portable manifest.
    (root/'model_bundle.json').write_text(json.dumps(report['manifest']))
    files = runtime.model_files(root)
    assert runtime.check_installed_manifest(root, files)['status'] == 'verified_files'
    manifest = report['manifest']
    manifest['timing_schema'] = 'old'
    (root/'model_bundle.json').write_text(json.dumps(manifest))
    with pytest.raises(runtime.PreflightError) as error:
        runtime.check_installed_manifest(root, files)
    assert error.value.code == 'model_manifest'


def test_real_models_load_one_at_a_time_on_cpu(inputs, tmp_path):
    _, root = inputs
    result = runtime.probe_models(root, tmp_path)
    assert result['audio']['label_source'] == 'chart_onset'
    assert result['4k']['decoder_positions'] == 4
    assert result['freex']['decoder_positions'] == 64


@pytest.mark.parametrize('broken', ['legacy', 'wrong_route', 'corrupt'])
def test_bad_models_fail_before_service_start(inputs, tmp_path, broken):
    _, root = inputs
    path = root/'decoder_freex/decoder_best.pt'
    if broken == 'legacy':
        checkpoint = torch.load(path, weights_only=True)
        checkpoint['timing_schema'] = 'legacy'
        torch.save(checkpoint, path)
        expected = UserWarning
    elif broken == 'wrong_route':
        path.write_bytes((root/'decoder/decoder_best.pt').read_bytes())
        expected = ValueError
    else:
        path.write_bytes(b'broken checkpoint')
        expected = Exception
    with pytest.raises(expected):
        runtime.probe_models(root, tmp_path)


def test_media_probe_uses_real_decode_without_optional_plotting(tmp_path):
    # A fresh interpreter blocks development-only packages, including plotting.
    code = '''
import importlib.abc, json, sys
from pathlib import Path
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib','cv2','torchvision','transformers','pytest'}:
            raise ImportError('development-only dependency: '+fullname)
sys.meta_path.insert(0, BlockOptional())
import torch
torch.set_num_threads(2)
from src.product import pipeline, runtime
result = runtime.probe_media(Path(sys.argv[1]))
assert result['wav']['mel_shape'][0] == 128
assert result['ogg']['mel_shape'][0] == 128
assert result['tempo']['status'] == 'executed'
assert 'matplotlib' not in sys.modules
print(json.dumps(result))
'''
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                             cwd=runtime.REPO_ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert abs(result['wav']['decoded_minus_source_seconds']) <= 1/22050
    assert 'decoded_minus_source_seconds' in result['ogg']  # do not hide padding
    assert not list(tmp_path.iterdir())


def test_cuda_failure_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(runtime.PreflightError) as error:
        runtime.probe_device('cuda')
    assert error.value.code == 'cuda_unavailable'


@pytest.mark.parametrize('requested,index', [('cuda',2), ('cuda:1',1)])
def test_requested_visible_cuda_selection_is_preserved(monkeypatch, requested, index):
    selected = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,7,8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda fraction, dev: selected.append((fraction,dev.index)))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda d: 'mock CUDA')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: (8,6))
    monkeypatch.setattr(torch.cuda, 'get_arch_list', lambda: ['sm_86'])
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda d: None)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda d: (8*runtime.GIB,12*runtime.GIB))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    real_arange = torch.arange
    monkeypatch.setattr(torch, 'arange', lambda *a, **k: real_arange(*a, **dict(k,device='cpu')))
    report = runtime.probe_device(requested)
    assert report['resolved'] == f'cuda:{index}'
    assert selected == [(.65,index)] and os.environ['CUDA_VISIBLE_DEVICES'] == '3,7,8'


def test_bad_cuda_index_reports_visible_device_issue(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(runtime.PreflightError) as error:
        runtime.probe_device('cuda:1')
    assert error.value.code == 'cuda_index'


def test_port_and_directory_lease_failures_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, 'which', lambda n: '/usr/bin/git')
    class OccupiedSocket:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def bind(self, address): raise OSError('address in use')
    monkeypatch.setattr(runtime.socket, 'socket', OccupiedSocket)
    with pytest.raises(runtime.PreflightError) as error:
        runtime.basic_environment(tmp_path, 7860)
    assert error.value.code == 'port_unavailable'
    import fcntl
    with (tmp_path/'.server.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(runtime.PreflightError) as error:
            runtime.basic_environment(tmp_path,7860)
    assert error.value.code == 'server_active'


def test_check_and_failed_preflight_do_not_exec_or_change_gpu_mapping(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5')
    monkeypatch.setattr(runtime.os, 'execv', lambda *a: pytest.fail('unexpected service start'))
    monkeypatch.setattr(runtime, 'preflight', lambda *a: dict(status='ready'))
    assert runtime.main(['--check','--output',str(tmp_path/'out')]) == 0
    monkeypatch.setattr(runtime, 'preflight', lambda *a: dict(status='failed'))
    assert runtime.main(['--output',str(tmp_path/'out')]) == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5'


def test_start_preserves_space_paths_and_exec_releases_probe(tmp_path, monkeypatch):
    model, output = tmp_path/'models with spaces', tmp_path/'output with spaces'
    calls = []
    monkeypatch.setattr(runtime, 'preflight', lambda *a: dict(status='ready',checks=dict(device=dict(resolved='cuda:0'))))
    monkeypatch.setattr(runtime.os, 'chdir', lambda path: calls.append(('cwd',path)))
    monkeypatch.setattr(runtime.os, 'execv', lambda executable, command: calls.append(('exec',executable,command)))
    assert runtime.main(['--model-root',str(model),'--output',str(output),'--device','cuda','--port','7890']) == 0
    assert calls[0] == ('cwd',runtime.REPO_ROOT)
    assert calls[1][1] == sys.executable
    command = calls[1][2]
    assert command[2] == 'script.serve_product'
    assert str(model) in command and str(output) in command and 'cuda:0' in command
    assert 'cuda' not in command
    assert 'train_models' not in ' '.join(command)
