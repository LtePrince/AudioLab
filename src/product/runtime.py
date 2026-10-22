"""Standard-library bootstrap; heavy dependencies are imported after preflight."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
import warnings

from .limits import (DEFAULT_MIN_AVAILABLE_RAM_GIB, ram_reserve_argument,
                     validate_ram_reserve)

SCHEMA = 'audiolab-linux-runtime-v1'
REPO_ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
MODEL_FILES = {
    'audio': 'transcriber/transcriber_best.pt',
    '4k': 'decoder/decoder_best.pt',
    'freex': 'decoder_freex/decoder_best.pt',
}
MIN_DISK_GIB = 8
MAX_RSS_GIB = 4
GPU_FRACTION = 0.65


class PreflightError(RuntimeError):
    def __init__(self, code, message, action):
        super().__init__(message)
        self.code, self.action = code, action


def fail(code, message, action):
    raise PreflightError(code, message, action)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def dependency_versions(requirements):
    """No import of torch, NumPy or any optional packages on this path."""
    result, missing, incompatible = {}, [], []
    for line in requirements.read_text().splitlines():
        line = line.partition('#')[0].strip()
        if not line:
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        matches = actual.split('+')[0] == expected
        result[name] = dict(installed=actual, baseline=expected, matches_baseline=matches)
        if not matches and name in ('torch', 'torchaudio', 'torchcodec'):
            incompatible.append(f'{name}={actual} (需要 {expected})')
    if missing:
        fail('dependencies_missing', '缺少运行依赖：'+', '.join(missing),
             '在独立环境按 docs/linux_runtime.md 安装 PyTorch，再安装 requirements/runtime.txt。')
    if incompatible:
        fail('torch_versions', 'PyTorch 组件版本不匹配：'+'; '.join(incompatible),
             '使用配套的 torch/torchaudio 2.10.0 与 torchcodec 0.10.0；CUDA 构建后缀可以不同于开发机。')
    return result


def resource_state(path):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':', 1)
        memory[key] = int(rest.split()[0])*1024
    rss = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    return dict(disk_free_gib=shutil.disk_usage(path).free/GIB,
                ram_available_gib=memory['MemAvailable']/GIB, rss_gib=rss/GIB)


def check_resources(path, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    min_available_ram_gib = validate_ram_reserve(min_available_ram_gib)
    state = resource_state(path)
    # Reserve room for the preflight report, small media fixtures, and service state.
    if state['disk_free_gib'] < MIN_DISK_GIB + 64*1024**2/GIB:
        fail('disk_reserve', '可用磁盘不足，需保留 8 GiB 并留出运行空间。',
             '清理自己的旧输出或用 --output 指定容量充足的磁盘；不要删除正在运行的任务。')
    if state['ram_available_gib'] < min_available_ram_gib or state['rss_gib'] > MAX_RSS_GIB:
        fail('ram_reserve', f'当前可用内存／进程内存超过运行预算：{state}',
             f'关闭占用内存的程序后重试；可用 RAM 至少 {min_available_ram_gib:g} GiB，工作进程 RSS 上限 4 GiB。')
    return state


def basic_environment(output, port, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    if platform.system() != 'Linux':
        fail('platform', '此运行入口面向 Linux。', '请在 Linux PC 中运行，其他系统尚未验证。')
    if sys.version_info < (3, 13):
        fail('python_version', '运行基线需要 Python 3.13 或更新版本。',
             '创建独立 Python 3.13 环境；当前测试基线为 3.13。')
    if shutil.which('git') is None:
        fail('git_missing', '未找到 git，生成报告需要记录代码版本。', '安装 git 并用 git clone 获取代码。')
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    check_resources(ancestor, min_available_ram_gib)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output) as stream:
        stream.write(b'AudioLab preflight\n')
        stream.flush()
    # The server still owns the authoritative directory lease and bind attempt.
    import fcntl
    with (output/'.server.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail('server_active', '该输出目录已有运行中的 AudioLab 服务。',
                 '使用已有服务，或为新实例选择另一个 --output 和 --port。')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    except OSError as error:
        fail('port_unavailable', f'本机端口 {port} 不可用：{error}', '选择空闲的 --port；服务只绑定 127.0.0.1。')


def model_files(model_root):
    result = {}
    for role, relative in MODEL_FILES.items():
        path = model_root/relative
        if not path.is_file():
            fail('model_missing', f'缺少权重：{relative}',
                 '先用 python -m script.model_bundle install 导入模型包，或指定正确的 --model-root。')
        if not 0 < path.stat().st_size <= 256*1024**2:
            fail('model_size', f'权重为空或超过此基线的 256 MiB 单文件上限：{relative}',
                 '重新校验并导入 timing-v1 模型包。')
        result[role] = dict(path=relative, **fingerprint(path))
    return result


def check_installed_manifest(model_root, files):
    path = model_root/'model_bundle.json'
    if not path.exists():
        return dict(status='unbundled', note='未提供模型包 manifest；已记录文件哈希，稍后实际加载结构。')
    from src.product.model_bundle import verify_installed
    try:
        manifest = verify_installed(model_root)
    except (ValueError, OSError) as error:
        fail('model_manifest', f'模型包校验失败：{error}',
             '重新校验并安装完整模型包，不混用旧文件。')
    return dict(status='verified_files', bundle_id=manifest.get('bundle_id'),
                manifest_sha256=fingerprint(path)['sha256'],
                note='manifest 仅证实文件一致性；真实模型结构另检。')


def probe_media(output):
    """Exercise the exact product decode/resample/mel path, including FFmpeg."""
    import numpy as np
    import soundfile as sf
    import torch
    from src.audio_understanding.inference import load_audio
    from src.product.tempo import estimate_tempo

    # 0.25 s at 48 kHz requires actual resampling to the model's 22050 Hz.
    samples = (0.1*np.sin(2*np.pi*440*np.arange(12000)/48000)).astype('float32')
    result = {}
    with tempfile.TemporaryDirectory(prefix='.media-check-', dir=output) as folder:
        for suffix, subtype in [('wav', 'PCM_16'), ('ogg', 'VORBIS')]:
            path = Path(folder)/f'probe.{suffix}'
            sf.write(path, samples, 48000, subtype=subtype)
            mel, duration = load_audio(path)
            if mel.shape[0] != 128 or not torch.isfinite(mel).all() or not .2 <= duration <= .3:
                raise RuntimeError(f'{suffix} decode/resampling/mel contract failed')
            if suffix == 'wav' and abs(duration-.25) > 1/22050:
                raise RuntimeError('WAV resampling changed the sample clock')
            # Vorbis decoders can expose end padding on very short files. This
            # probe detects working native media libraries, not PGR exportability
            # of every OGG encoding; packaging retains its strict duration check.
            result[suffix] = dict(source_duration_seconds=.25, decoded_duration_seconds=duration,
                                  decoded_minus_source_seconds=duration-.25, mel_shape=list(mel.shape))
    # Exercise librosa's lazy tempo/Numba/SciPy path on a small controlled input.
    periodic = np.zeros((128, 344), dtype='float32')
    periodic[:, ::22] = 12
    estimate = estimate_tempo(periodic, 344*512/22050)
    if estimate['status'] != 'estimated_unconfirmed':
        raise RuntimeError('tempo library probe failed')
    result['tempo'] = dict(status='executed', method=estimate['method'],
                           quality_validation=False)
    result['libsndfile'] = sf.__libsndfile_version__
    return result


def probe_models(model_root, output, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    import torch
    from src.audio_understanding.inference import load_model
    from src.product.experimental_models import load_models, runtime_contract

    result = {}
    for role, relative in MODEL_FILES.items():
        check_resources(output, min_available_ram_gib)
        # Legacy checkpoints are still loadable by research tools; this runtime
        # baseline refuses them rather than silently mixing timing conventions.
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Checkpoint uses legacy/unknown timing.*')
            if role == 'audio':
                model, metadata = load_model(model_root/relative, 'cpu')
                result[role] = dict(status='loaded', label_source=metadata['label_source'])
                del model
            else:
                contract = runtime_contract(model_root/relative,
                    audio_sha256=fingerprint(model_root/MODEL_FILES['audio'])['sha256'], mode=role)
                encoder, decoder = load_models(str(model_root/relative), torch.device('cpu'))
                expected = 4 if role == '4k' else 64
                if decoder.n_lanes != expected or encoder.conv_in.in_channels != 128 or encoder.mel_per_frame != 2:
                    raise ValueError(f'{role}: incompatible model configuration')
                result[role] = dict(status='loaded', decoder_positions=decoder.n_lanes)
                if contract is not None:
                    result[role].update(runtime_contract=contract, quality_status='experimental')
                del encoder, decoder
        gc.collect()
        check_resources(output, min_available_ram_gib)
    return result


def probe_device(requested):
    import torch
    device = torch.device(requested)
    result = dict(requested=requested, torch=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            fail('cuda_unavailable', 'PyTorch 无法使用 NVIDIA CUDA。',
                 '检查 nvidia-smi、驱动、CUDA_VISIBLE_DEVICES 以及所装 PyTorch CUDA wheel；不会自动改为 CPU。')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            fail('cuda_index', f'不可见的 CUDA 设备索引：{index}',
                 '用可见设备索引选择 --device cuda:0；CUDA_VISIBLE_DEVICES 会重新编号。')
        device = torch.device('cuda', index)
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, device)
        result.update(name=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)),
                      compiled_arches=torch.cuda.get_arch_list())
    with torch.inference_mode():
        x = torch.arange(16, dtype=torch.float32, device=device).reshape(4,4)
        value = (x @ x.T).sum().item()
        if value != 3680:
            raise RuntimeError('device arithmetic check failed')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(free_gib=free/GIB, total_gib=total/GIB, memory_fraction=GPU_FRACTION)
        del x
        torch.cuda.empty_cache()
    result.update(resolved=str(device), status='small_kernel_passed')
    return result


def preflight(model_root, output, device, port, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    start = time.monotonic()
    report = dict(schema=SCHEMA, status='checking', python=platform.python_version(),
                  system=platform.platform(), model_root=str(model_root), output=str(output),
                  port=port, device_requested=device, checks={}, warnings=[],
                  limitations=['预检不是整曲生成、质量或目标 PC 实测认证。'])
    phase = 'environment'
    try:
        min_available_ram_gib = validate_ram_reserve(min_available_ram_gib)
        report['resource_limits'] = dict(min_available_ram_gib=min_available_ram_gib,
            min_free_disk_gib=MIN_DISK_GIB, max_rss_gib=MAX_RSS_GIB, gpu_memory_fraction=GPU_FRACTION)
        basic_environment(output, port, min_available_ram_gib)
        report['checks']['resources_before'] = check_resources(output, min_available_ram_gib)
        phase = 'dependencies'
        report['checks']['dependencies'] = dependency_versions(REPO_ROOT/'requirements/runtime.txt')
        if any(not value['matches_baseline'] for value in report['checks']['dependencies'].values()):
            report['warnings'].append('部分数值依赖与固定基线不同；记录实际版本，不代表该组合已完成整曲回归。')
        if sys.version_info[:2] != (3,13):
            report['warnings'].append('Python 与已测试的 3.13 不同。')
        phase = 'model_files'
        report['checks']['files'] = model_files(model_root)
        report['checks']['bundle'] = check_installed_manifest(model_root, report['checks']['files'])
        phase = 'imports'
        import torch
        torch.set_num_threads(2)
        import src.product.pipeline  # actual runtime import closure, no dataset/cache loading
        check_resources(output, min_available_ram_gib)
        phase = 'media'
        report['checks']['media'] = probe_media(output)
        phase = 'models'
        report['checks']['models'] = probe_models(model_root, output, min_available_ram_gib)
        phase = 'device'
        report['checks']['device'] = probe_device(device)
        report['checks']['resources_after'] = check_resources(output, min_available_ram_gib)
        report['status'] = 'ready'
    except Exception as error:
        actions = dict(imports='按 docs/linux_runtime.md 安装配套依赖与 FFmpeg 共享库，再运行预检。',
                       media='检查 FFmpeg 共享库和 WAV／OGG 解码支持；命令行 ffmpeg 可用不代表 TorchCodec 已能加载。',
                       models='用可信模型包重新安装三份配套权重；不要混用旧时钟或其他模型。',
                       device='检查 NVIDIA 驱动与所装 PyTorch CUDA 构建，选择受支持的 GPU。')
        report.update(status='failed', error=dict(phase=phase, code=getattr(error,'code',phase+'_failed'),
                       type=type(error).__name__, message=str(error),
                       action=getattr(error,'action',actions.get(phase, '检查路径、权限和 docs/linux_runtime.md 后重试。'))))
    report['elapsed_seconds'] = time.monotonic()-start
    return report


def server_command(model_root, output, device, port, min_available_ram_gib=DEFAULT_MIN_AVAILABLE_RAM_GIB):
    min_available_ram_gib = validate_ram_reserve(min_available_ram_gib)
    return [sys.executable, '-m', 'script.serve_product', '--model-root', str(model_root),
            '--output', str(output), '--device', device, '--port', str(port),
            '--min-available-ram-gib', str(min_available_ram_gib)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='preflight only; do not start HTTP service')
    parser.add_argument('--model-root', type=Path, default=Path('checkpoints/timing_v1'))
    parser.add_argument('--output', type=Path, default=Path('out/local-service'))
    parser.add_argument('--device', default='cuda', help='cuda (default), cuda:N, or explicit cpu')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--min-available-ram-gib', type=ram_reserve_argument,
                        default=DEFAULT_MIN_AVAILABLE_RAM_GIB,
                        help='system available RAM reserve; default 8 GiB, minimum 1 GiB')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'cpu|cuda(?::\d+)?', args.device):
        parser.error('--device must be cpu, cuda, or cuda:N')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    model_root, output = args.model_root.resolve(), args.output.resolve()
    if output == model_root or model_root in output.parents or output in model_root.parents:
        parser.error('--output and --model-root must be separate directories')
    # Set only task-specific defaults; preserve the user's visible GPU mapping.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    try:
        report = preflight(model_root, output, args.device, args.port, args.min_available_ram_gib)
    except KeyboardInterrupt:
        return 130
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report['status'] != 'ready':
        return 1
    if args.check:
        return 0
    # Reuse the exact visible device that passed the probe in this environment.
    command = server_command(model_root, output, report['checks']['device']['resolved'],
                             args.port, args.min_available_ram_gib)
    print(f'AudioLab: http://127.0.0.1:{args.port} （Ctrl+C 停止）', flush=True)
    os.chdir(REPO_ROOT)
    # Replacing the process releases all preflight models/CUDA state before
    # the lightweight HTTP server starts spawning bounded inference workers.
    os.execv(sys.executable, command)
    return 0
