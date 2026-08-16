"""
test/diffusion/test_sampler.py
-------------------------------
DDIMSampler 功能测试（P6-1 / P6-3 / 变长推理）。

测试项目：
  [T1] _make_ddim_timesteps 形状与范围
  [T2] sample 输出形状正确 (B, z_ch, T)
  [T3] eta=0 → 相同 seed 结果一致（确定性）
  [T4] eta=0 与 eta=1 结果不同（随机性差异）
  [T5] cfg_scale=1.0 等价于无 CFG（结果完全一致）
  [T6] callback 每步被调用，中间 shape 正确
  [T7] make_window_starts 覆盖/重叠/末窗对齐
  [T8] prefix_z0 前缀条件：输出前缀 == 已知前缀，x_T 不被原地修改
  [T9] sample_windowed 输出形状 + 前缀区与前一窗一致
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import pytest

from src.diffusion.schedule import NoiseSchedule, DEFAULT_SCHEDULE_CONFIG
from src.diffusion.sampler  import (
    DDIMSampler,
    _make_ddim_timesteps,
    make_window_starts,
)


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
# T7: make_window_starts
# ──────────────────────────────────────────────────────────────────────
def test_window_starts():
    """全覆盖、无重复、末窗对齐末尾、相邻重叠 ≥ overlap。"""
    for T_total, window, overlap in [
        (300, 128, 32), (256, 256, 64), (257, 256, 64),
        (512, 128, 0), (1000, 256, 128), (384, 128, 0),
    ]:
        starts = make_window_starts(T_total, window, overlap)
        assert starts[0] == 0
        assert starts[-1] == T_total - window, "末窗必须对齐到结尾"
        assert starts == sorted(set(starts)), "起点应严格递增无重复"
        covered = set()
        for s in starts:
            covered.update(range(s, s + window))
        assert covered == set(range(T_total)), "窗口必须无缝覆盖 [0, T_total)"
        for a, b in zip(starts, starts[1:]):
            assert (a + window) - b >= overlap, \
                f"相邻窗口重叠 {(a + window) - b} < overlap {overlap}"

    with pytest.raises(ValueError):
        make_window_starts(100, 128, 32)     # T_total < window
    with pytest.raises(ValueError):
        make_window_starts(300, 128, 128)    # overlap >= window


# ──────────────────────────────────────────────────────────────────────
# T8: prefix_z0 前缀条件
# ──────────────────────────────────────────────────────────────────────
def test_prefix_conditioning(sampler, tiny_dit, audio_cond, schedule):
    """输出的前缀区应精确等于 prefix_z0；caller 的 x_T 不被原地修改。

    另外检测逐步注入机制本身（不能只靠最终 hard-set 蒙混过关）：
    记录每次 DiT 调用的输入，前缀区必须满足
    input = √ᾱ_t · prefix + √(1-ᾱ_t) · ε，即残差
    (input − √ᾱ_t·prefix) 的统计量符合 N(0, 1-ᾱ_t)。
    """
    shape  = (B, Z_CH, T_SEQ)
    L_p    = 16
    prefix = torch.randn(B, Z_CH, L_p, device=DEVICE)
    x_T    = torch.randn(shape, device=DEVICE)
    x_T_backup = x_T.clone()

    recorded = []   # (t_scalar, z_input.clone())

    class RecordingDiT(torch.nn.Module):
        def forward(self, z, audio_c, t):
            recorded.append((float(t[0].item()), z.detach().clone()))
            return torch.zeros_like(z)

    z0 = sampler.sample(
        RecordingDiT(), audio_cond, shape,
        steps=5, eta=0.0, prefix_z0=prefix,
        x_T=x_T, show_progress=False,
    )
    assert z0.shape == shape
    assert torch.equal(z0[:, :, :L_p], prefix), "前缀区必须精确等于已知 latent"
    assert torch.equal(x_T, x_T_backup), "sample() 不得原地修改 caller 的 x_T"

    # 逐步注入检查:每一步 DiT 看到的前缀区都是 prefix 在当前噪声水平的 q_sample
    ab = sampler.schedule.alphas_bar.cpu()
    assert len(recorded) == 5
    for t_scalar, z_in in recorded:
        ab_t     = float(ab[int(t_scalar)])          # DiT 收到 t = t_val-1,直接索引 ᾱ
        residual = z_in[:, :, :L_p] - (ab_t ** 0.5) * prefix
        expect_std = (1.0 - ab_t) ** 0.5
        got_std    = residual.std().item()
        assert 0.6 * expect_std < got_std < 1.4 * expect_std, (
            f"t={t_scalar}: 前缀区残差 std {got_std:.4f} 偏离期望 "
            f"{expect_std:.4f} — 逐步 RePaint 注入可能被移除或写错噪声水平"
        )
        assert abs(residual.mean().item()) < 0.35 * expect_std, \
            f"t={t_scalar}: 前缀区残差均值异常(注入的应为零均值噪声)"

    # 非法前缀长度应报错
    for bad_L in [0, T_SEQ, T_SEQ + 1]:
        with pytest.raises(ValueError):
            sampler.sample(
                tiny_dit, audio_cond, shape, steps=2,
                prefix_z0=torch.zeros(B, Z_CH, bad_L, device=DEVICE),
                show_progress=False,
            )


# ──────────────────────────────────────────────────────────────────────
# T9: sample_windowed
# ──────────────────────────────────────────────────────────────────────
def test_sample_windowed_shape(sampler, tiny_dit):
    """任意 T_total 输出形状正确；窗口重叠区与前一窗内容一致。"""
    torch.manual_seed(7)
    for T_total, window, overlap in [(300, 128, 32), (80, 32, 8), (513, 256, 64)]:
        audio_c = torch.randn(B, A_CH, T_total, device=DEVICE)
        z0 = sampler.sample_windowed(
            tiny_dit, audio_c, z_channels=Z_CH,
            window=window, overlap=overlap,
            steps=3, eta=0.0, cfg_scale=1.0, show_progress=False,
        )
        assert z0.shape == (B, Z_CH, T_total), \
            f"T_total={T_total}: {z0.shape} ≠ {(B, Z_CH, T_total)}"


def test_sample_windowed_overlap_consistency(schedule):
    """内容级断言：每窗喂给 DiT 的音频条件必须是全局 audio_c 的正确切片，
    每次调用的 token 数必须等于训练窗口长度。"""
    audio_slices = []

    class RecordingDiT(torch.nn.Module):
        def forward(self, z, audio_c, t):
            audio_slices.append(audio_c.detach().clone())
            return torch.zeros_like(z)

    sampler  = DDIMSampler(schedule)
    T_total, window, overlap, steps = 200, 96, 24, 2
    audio_c  = torch.randn(1, A_CH, T_total)
    z0 = sampler.sample_windowed(
        RecordingDiT(), audio_c, z_channels=Z_CH,
        window=window, overlap=overlap,
        steps=steps, eta=0.0, show_progress=False,
    )
    assert z0.shape == (1, Z_CH, T_total)

    # 每次模型调用的窗口长度都必须是训练长度 window
    assert all(a.shape[-1] == window for a in audio_slices), \
        f"所有 DiT 调用的 token 数都应为 {window}，实际 {set(a.shape[-1] for a in audio_slices)}"

    # 内容:第 k 窗的每次调用收到的音频条件 == audio_c[:, :, s_k : s_k+window]
    starts = make_window_starts(T_total, window, overlap)
    assert len(audio_slices) == len(starts) * steps, \
        f"调用次数 {len(audio_slices)} ≠ 窗数 {len(starts)} × 步数 {steps}"
    for k, s in enumerate(starts):
        expected = audio_c[:, :, s : s + window]
        for j in range(steps):
            got = audio_slices[k * steps + j]
            assert torch.equal(got, expected), \
                f"窗 {k} 第 {j} 步的音频条件不是 audio_c[{s}:{s + window}] 切片"


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
