"""Resource checks and atomic, space-checked training checkpoints."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import torch

GIB = 1024 ** 3


def resource_snapshot(path: str | Path) -> dict[str, float]:
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        info[key] = int(value.split()[0]) * 1024
    rss = 0
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss = int(line.split()[1]) * 1024
    return dict(disk_free_gib=shutil.disk_usage(path).free / GIB,
                ram_available_gib=info['MemAvailable'] / GIB,
                rss_gib=rss / GIB)


def add_resource_args(parser) -> None:
    g = parser.add_argument_group('resource limits')
    g.add_argument('--min-free-disk-gib', type=float, default=8)
    g.add_argument('--min-available-ram-gib', type=float, default=8)
    g.add_argument('--max-rss-gib', type=float, default=10)
    g.add_argument('--gpu-memory-fraction', type=float, default=0.65)
    g.add_argument('--keep-checkpoints', type=int, default=1)


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class ResourceBudget:
    def __init__(self, args, device: torch.device, directory: Path):
        self.args, self.device, self.directory = args, device, directory
        self._last_check = 0.0
        if not 0 < args.gpu_memory_fraction <= 1:
            raise ValueError('gpu-memory-fraction must be in (0, 1]')
        if args.keep_checkpoints < 1:
            raise ValueError('keep-checkpoints must be at least 1 for recovery')
        if device.type == 'cuda':
            # PyTorch's allocator limit requires an explicit CUDA index, unlike
            # tensor.to('cuda'). Resolve the current visible device, not GPU 0.
            if device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, self.device)
        self.check(force=True)

    def check(self, *, force: bool = False, extra_bytes: int = 0) -> dict | None:
        if not force and time.monotonic() - self._last_check < 5:
            return None
        self._last_check = time.monotonic()
        state = resource_snapshot(self.directory)
        if state['disk_free_gib'] < self.args.min_free_disk_gib + extra_bytes / GIB:
            raise RuntimeError(f'Resource budget: insufficient disk reserve: {state}')
        if state['ram_available_gib'] < self.args.min_available_ram_gib:
            raise RuntimeError(f'Resource budget: insufficient available RAM: {state}')
        if state['rss_gib'] > self.args.max_rss_gib:
            raise RuntimeError(f'Resource budget: process RSS limit exceeded: {state}')
        return state

    def save(self, payload: dict, path: str | Path) -> None:
        """Keep the previous checkpoint intact until the new file is complete.

        Check free space for the full temporary file, even on replacement.
        Checkpoint rotation happens only after this method succeeds.
        """
        path = Path(path)
        self.check(force=True, extra_bytes=int(tensor_bytes(payload) * 1.1) + 16 * 1024 ** 2)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as fh:
                torch.save(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def log_state(self) -> dict:
        state = resource_snapshot(self.directory)
        if self.device.type == 'cuda':
            state['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated(self.device) / GIB
            state['gpu_peak_reserved_gib'] = torch.cuda.max_memory_reserved(self.device) / GIB
        return state
