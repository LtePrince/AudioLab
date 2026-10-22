# 旧 diffusion 与 AR：同一 benchmark 对照（2026-10-22）

后续已收到历史权重 Release，并完成相同 8 例的[Release 权重复跑](diffusion_release_20261022.md)。下文保留此前本地权重对照，勿将两组权重混同。

用户反馈上传的旧 diffusion 谱节奏优于 AR。本轮对上传文件原样评分，并用本地现存旧 diffusion 权重执行相同 8 个案例。未重训、未改阈值、未吸附拍点、未改变 benchmark v1。用户随后确认历史权重可能在另一台 PC，正在查找，因此本地复跑不宣称复现上传谱。

## Eltaw 同曲结果

同一音频 SHA、同一参考谱的 701 个唯一起点、BPM=193、offset=0，冻结协议 SHA `5b5fb33e54aea677bca80c78889964b4904c564142f021ee587f66d9e4f4effd`。这里的 P/R/F1 是对参考谱取音的匹配，非人工声学真值，也不代表唯一正确编排。

| 产物 | 音符／唯一起点 | F1@25ms | F1@50ms | P@50ms | R@50ms | 网格奖励/起点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 上传的 diffusion v2 | 531/463 | 0.6821 | 0.7199 | 0.9050 | 0.5977 | 0.2505 |
| 已交付 AR global / seed42 | 1321/1183 | 0.5403 | 0.5403 | 0.4303 | 0.7261 | 1.0000 |
| 本地 diffusion 默认复跑 | 534/489 | 0.3092 | 0.4824 | 0.5869 | 0.4094 | 0.1636 |

上传旧谱在本曲 F1 和精确率上优于已交付 AR，支持用户对这份旧谱的评价；AR 召回更高，但多出了大量不匹配参考取音的起点。上传谱未匹配参考起点的输出数为 44，AR 为 674，均为 ±50ms 口径；这不是对这些额外编排的逐一人工否定。

网格奖励仍严格只加分。193 BPM 下 1 tick 约 9.715ms；上传谱相对参考的已匹配起点绝对误差中位数约 9.715ms、P95 29.145ms（±100ms 配对），不少音符落在所选格点附近但未精确命中。AR 网格满分仍可能过密／取音不当。本轮不因旧谱得分低而事后放宽网格宽度，不将该项当作模型排名。

键型数量：上传谱 Tap/Drag/Hold/Flick=252/242/35/2；AR=1189/12/111/9；本地复跑=238/260/35/1。类型占比没有目标值，不因更接近某比例就判优。上传谱最后一个音符约 130.560s，AR 为 127.306s，音频为 135.551s。

上传谱的未知后处理无法还原；AR benchmark 使用原始输出，与实际试玩包起点相同，但试玩包另截短了 19 个 Hold。本表不据两者最终几何差异直接归因模型。

## 本地 diffusion 权重的完整 8 例复跑

固定现存 `checkpoints/dit/dit_best.pt`（epoch 297）、与其内嵌 Wave 完全相同的 `wave_finetuned.pt`、`checkpoints/vae/vae_final.pt`（latent_scale≈1.0040），seed=42、DDIM=50、CFG=3、eta=0，当前 src/test.py 原生长度推理。沿用 logits>0 的二值解释、旧 converter 的 Hold 解码；未额外修复或调参。旧权重没有现行 timing_schema，已记录 legacy/unknown；加载警告不阻止本次对照。

| 案例 | 唯一起点 | F1@50ms | 结果说明 |
| --- | ---: | ---: | --- |
| piano_three | 37 | 0.1500 | 基础检查失败 |
| piano_weak_middle | 22 | 0.1600 | 基础检查失败 |
| piano_repeated | 31 | 0.1176 | 基础检查失败 |
| piano_chunk_boundary | 123 | 0.0476 | 基础检查失败 |
| silence | 23 | 0.0000 | 纯静音误生成 25 个音符／23 个唯一起点 |
| -SURREALISM-_IN | 627 | 0.4023 | 开发曲诊断；非质量通过 |
| ABYSS MOTION 192.333_IN | 582 | 0.4705 | 开发曲诊断；非质量通过 |
| Eltaw_IN | 489 | 0.4824 | 开发曲诊断；非质量通过 |

本地 diffusion 在三个开发曲的 F1@50ms 宏平均为 0.4517，AR global/seed42 为 0.6037。三音例找到了 3 个参考起点，但输出共 37 个唯一起点；弱音例漏掉弱音；纯静音也出音。说明现存这组权重尚不能作为可用模型交付，也没有复现上传谱的 Eltaw 0.7199。不能从这一组未知历史对应关系的复跑推广为“diffusion 架构不如 AR”。

AR 仍缺重复音／跨块两例；上传包只有 Eltaw 一曲。两者缺项保留在报告，不按通过计算。本轮继续使用已有开发数据，不新增独立测试集。旧 diffusion 的原始训练划分未重新核定；JSON 中 seen_training 沿用 AR 基准案例标签，不证明每首都参加过该旧 DiT 的训练。

## 复现、资源与下一步

已新增 `script/benchmark_diffusion.py`，从原 benchmark inputs 读取同一案例，核对音频 SHA，分别记录上传谱和本地模型；所有小样例、曲目逐项写入固定评分器。上传 ZIP 只读取所需成员，不修改、解压或复制整包。

```bash
python -m script.train_models --gpu 1 --worker script.benchmark_diffusion -- \
  --output out/diffusion_benchmark_new \
  --upload /path/to/Eltaw_4k_generate_v2.zip \
  --min-available-ram-gib 4 --max-rss-gib 4
```

可以用 `--dit`、`--vae`、`--wave`、`--seed`、`--cfg-scale`、`--ddim-steps`、`--fixed-length` 明确换入找回的历史配置。先确认旧 DiT/VAE/Wave 三份权重及生成命令，再按完全相同协议重跑；当前未进行参数搜索或重训。

23 项适配器与 benchmark 测试通过。8 次推理累计 11.63 秒；进程 RSS 快照 1.509 GiB，torch GPU 峰 allocated/reserved=0.345/0.447 GiB。仅为本机单次测量。

完整本地产物：`out/diffusion_benchmark_20261022/model/inputs.json`、`report.json`、8 份生成 JSON 及逐例日志。Git 保留本摘要和来源指纹，原始音频／权重／完整预测不入库。
