"""
test/diffusion/test_sampler.py
-------------------------------
DDIMSampler 功能测试（P6-1 / P6-3）。

测试项目：
  [T1] _make_ddim_timesteps 形状与范围
  [T2] sample 输出形状正确 (B, z_ch, T)
  [T3] eta=0 → 相同 seed 结果一致（确定性）
  [T4] eta=0 与 eta=1 结果不同（随机性差异）
  [T5] cfg_scale=1.0 等价于无 CFG（结果完全一致）
  [T6] callback 每步被调用，中间 shape 正确
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import pytest

from src.diffusion.schedule import NoiseSchedule, DEFAULT_SCHEDULE_CONFIG
from src.diffusion.sampler  import DDIMSampler, _make_ddim_timesteps


# ──────────────────────────────────────────────────────────────────────
# 极小模型：仅用于跑通流程
# ──────────────────────────────────────────────────────────────────────
class TinyDiT(torch.nn.Module):
    """始终输出与输入相同形状的零张量，充当 DiT 占位符。"""
    def forward(self, z, audio_c, t):   # (B,C,L), (B,A,L), (B,)
        return torch.zeros_like(z)


DEVICE = "cpu"
B, Z_CH, T_SEQ, A_CH = 2, 4, 32, 8   # 极小尺寸，测试速度 < 1s

@pytest.fixture(scope="module")
def schedule():
    cfg = {**DEFAULT_SCHEDULE_CONFIG, "T": 100}  # 缩短到 100 步
    ns  = NoiseSchedule(**cfg).to(DEVICE)
    return ns

@pytest.fixture(scope="module")
def sampler(schedule):
    return DDIMSampler(schedule)

@pytest.fixture(scope="module")
def tiny_dit():
    return TinyDiT().to(DEVICE)

@pytest.fixture(scope="module")
def audio_cond():
    torch.manual_seed(0)
    return torch.randn(B, A_CH, T_SEQ, device=DEVICE)


# ──────────────────────────────────────────────────────────────────────
# T1: _make_ddim_timesteps
# ──────────────────────────────────────────────────────────────────────
def test_ddim_timesteps_shape():
    """子步长度精确 = S，范围 [1, T]，降序。"""
    for S in [5, 10, 20]:
        ts = _make_ddim_timesteps(T=100, S=S)
        assert len(ts) == S, f"S={S} but len={len(ts)}"
        assert ts[0] > ts[-1],   "应降序"
        assert ts.min() >= 1,    "最小值 >= 1"
        assert ts.max() <= 100,  "最大值 <= T"


# ──────────────────────────────────────────────────────────────────────
# T2: 输出形状
# ──────────────────────────────────────────────────────────────────────
def test_sample_shape(sampler, tiny_dit, audio_cond):
    shape = (B, Z_CH, T_SEQ)
    z0 = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, cfg_scale=1.0, show_progress=False,
    )
    assert z0.shape == shape, f"期望 {shape}，得到 {z0.shape}"


# ──────────────────────────────────────────────────────────────────────
# T3: eta=0 确定性
# ──────────────────────────────────────────────────────────────────────
def test_sample_deterministic(sampler, tiny_dit, audio_cond):
    shape = (B, Z_CH, T_SEQ)
    noise = torch.randn(shape, device=DEVICE)

    z0_a = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, x_T=noise.clone(), show_progress=False,
    )
    z0_b = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, x_T=noise.clone(), show_progress=False,
    )
    assert torch.allclose(z0_a, z0_b), "eta=0 应为确定性，两次结果应完全相同"


# ──────────────────────────────────────────────────────────────────────
# T4: eta=1 引入随机性（两次不同 seed → 不同结果）
# ──────────────────────────────────────────────────────────────────────
def test_sample_stochastic(sampler, tiny_dit, audio_cond):
    shape = (B, Z_CH, T_SEQ)
    noise = torch.randn(shape, device=DEVICE)

    torch.manual_seed(42)
    z0_a = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=1.0, x_T=noise.clone(), show_progress=False,
    )
    torch.manual_seed(99)
    z0_b = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=1.0, x_T=noise.clone(), show_progress=False,
    )
    assert not torch.allclose(z0_a, z0_b), "eta=1 使用不同 seed 应产生不同结果"


# ──────────────────────────────────────────────────────────────────────
# T5: cfg_scale=1.0 等价于关闭 CFG（输出完全一致）
# ──────────────────────────────────────────────────────────────────────
def test_cfg_scale_one_equiv(sampler, tiny_dit, audio_cond):
    """TinyDiT 始终输出 0，故两路结果应完全相同。"""
    shape = (B, Z_CH, T_SEQ)
    noise = torch.randn(shape, device=DEVICE)

    z_no_cfg = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, cfg_scale=1.0,
        x_T=noise.clone(), show_progress=False,
    )
    z_cfg1 = sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, cfg_scale=1.0,
        audio_c_uncond=torch.zeros_like(audio_cond),
        x_T=noise.clone(), show_progress=False,
    )
    assert torch.allclose(z_no_cfg, z_cfg1), "cfg_scale=1.0 应等价于无 CFG"


# ──────────────────────────────────────────────────────────────────────
# T6: callback 调用次数与中间形状
# ──────────────────────────────────────────────────────────────────────
def test_callback(sampler, tiny_dit, audio_cond):
    shape = (B, Z_CH, T_SEQ)
    calls = []

    def cb(step_idx, x_t):
        calls.append((step_idx, x_t.shape))

    sampler.sample(
        tiny_dit, audio_cond, shape,
        steps=5, eta=0.0, callback=cb, show_progress=False,
    )
    assert len(calls) == 5, f"应有 5 次 callback，实际 {len(calls)}"
    for idx, sh in calls:
        assert sh == shape, f"step {idx}: 中间形状 {sh} 应为 {shape}"


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
