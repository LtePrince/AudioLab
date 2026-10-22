# Release 历史 diffusion 权重：同一 benchmark 复跑

已下载并校验 [diffusion-legacy-20261022](https://github.com/LtePrince/AudioLab/releases/tag/diffusion-legacy-20261022)，按发布参数完成原有 8 例推理。Eltaw F1@50ms 从此前本地权重的 0.4824 提升到 0.5197，但未复现用户上传旧谱的 0.7199。五个构造案例仍未通过基础检查。

## 权重与复现范围

Release DiT 为 epoch 293、12 层、384 维、64,495,120 参数；此前本地 DiT 为 epoch 297、8 层、256 维。此次 VAE 与本地版本哈希一致，Wave 则是随 Release 提供、与其 DiT 内嵌 Wave 逐 tensor 一致的版本。三套模型均已 strict 加载并完成推理。

| 文件 | SHA-256 |
| --- | --- |
| ZIP | `2b6bf62d7f9299b6af7bbb4c23d010225ecc80ebe1d358297c1ed9ae8f8cdf30` |
| DiT | `030dc32dd29d95d61a37df33e9a8717a48f7ed974296408bb465c5061c04afa5` |
| VAE | `885cfe7123deed8df30defd5b796002228f634da085d811af2c21e64538ba468` |
| Wave | `ff51ea2007efef99d1aa864062944d8cdafb40a558e9d66abe70e79c7285021c` |

发布说明中的历史 shell 命令引用 `dit_final.pt` 和 `checkpoints/wave/wave_final.pt`；前者尚未找回。Release 提供的是保存下来的 `dit_best.pt` + 联合微调 Wave。这组文件可用于对照，但尚无证据证明它就是上传 Eltaw v2 所用组合。CFG=3、DDIM=50 有历史命令依据；seed=42、eta=0、原生长度及窗口参数依据此次发布说明，不能当成已证实的旧谱生成参数。

此次生成路径中的 `src/test.py`、音频/谱面转换、DiT、VAE、Wave 和 diffusion 代码，与发布核对版本 `8c39320284b80c60f4256b085656c043908e3610` 无差异。旧权重缺少现行 timing_schema，仍记录为 legacy/unknown。没有重训、调阈值或搜索采样参数。

## 同一评分口径

固定 benchmark v1，协议 SHA `5b5fb33e54aea677bca80c78889964b4904c564142f021ee587f66d9e4f4effd`。8 个案例定义与上轮完全相同，逐一核对音频 SHA。参考 BPM/offset 只用于时间单位转换，参考音符不进入模型。

本表统计唯一起点与参考谱的时间匹配，不评价四键位置/键型是否正确，也不是全部声学 onset 的人工真值。网格项继续按 32 ticks/拍、0/4/8/16/24/32 相位、1 tick 容差加分；不改变输出，不作为固定排布要求或单独的质量通过条件。

| Eltaw 产物 | 音符/唯一起点 | F1@25ms | F1@50ms | P@50ms | R@50ms | 网格奖励/起点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 用户上传旧 diffusion v2 | 531/463 | 0.6821 | 0.7199 | 0.9050 | 0.5977 | 0.2505 |
| AR global / seed42 | 1321/1183 | 0.5403 | 0.5403 | 0.4303 | 0.7261 | 1.0000 |
| **Release diffusion / seed42** | **587/542** | **0.3073** | **0.5197** | **0.5959** | **0.4608** | **0.1734** |
| 此前本地 diffusion / seed42 | 534/489 | 0.3092 | 0.4824 | 0.5869 | 0.4094 | 0.1636 |

Release 本曲 Tap/Drag/Hold/Flick 为 242/293/51/1；首音 1.2241s，末音 130.6088s，音频长 135.5508s。该数量仅为诊断，无目标比例。用户上传谱的数量和时间序列均不同，因此没有复现成功；不能通过本轮结果否定用户上传旧谱优于 AR 的反馈。AR 原始输出与试玩包的起点相同，但试玩包另缩短了 19 个 Hold，几何差异不能直接归因于模型。

## 八例结果与缺口

| 案例 | Release 音符/唯一起点（含越界） | Release F1@50ms | 此前本地 diffusion F1@50ms |
| --- | ---: | ---: | ---: |
| piano_three | 60/51 | 0.1111 | 0.1500 |
| piano_weak_middle | 66/59 | 0.0968 | 0.1600 |
| piano_repeated | 62/50 | 0.0755 | 0.1176 |
| piano_chunk_boundary | 34/33 | 0.1111 | 0.0476 |
| silence | 47/41 | 0.0000 | 0.0000 |
| -SURREALISM-_IN | 674/565 | 0.3631 | 0.4023 |
| ABYSS MOTION 192.333_IN | 666/596 | 0.5004 | 0.4705 |
| Eltaw_IN | 587/542 | 0.5197 | 0.4824 |

三个自然曲目的 F1@50ms 宏平均：Release 0.4611，此前本地 diffusion 0.4517，AR 0.6037。只有一个采样 seed，DiT 规模、训练结果和 Wave 均不同，不能将差异单独归因于参数量。沿用已有开发曲，不新增独立歌曲；案例中 seen_training 是 AR 基准沿用的标签，不证明旧 DiT 的原训练划分。

- 三音例匹配到全部 3 个起点，但共生成 51 个唯一起点，其中 11 个处于标注静音区。
- 弱中音在 ±50ms 下被匹配到；该例共 59 个唯一起点，不能仅凭弱音召回变为 1 就判定改善。三个起点在 ±25ms 下均未匹配。
- 纯静音仍生成 47 个音符、41 个唯一起点；其中 40 个在音频内，另一个位于 10.015625s。
- 弱音例还在 10.0s 恰好产生一个起点。协议使用 `[0, duration)`，该点与纯静音末端点均被记为越界并计入误报，未静默截掉。
- SURREALISM 首音 11.9583s，参考首音 1.3333s；ABYSS 首音 13.7262s，参考首音 3.7435s。开头漏取音仍需处理。
- AR 缺重复音/跨块两例，上传谱仅有 Eltaw；完整报告继续保留这些缺项。自然曲通过格式检查不代表音乐质量通过。

下一步优先核对旧谱的原生成组合和时间处理，保持评分协议冻结；同时，静音出音、稀疏音过密和开头漏取音是已明确的训练/推理缺口。当前结果不足以选择 AR 或 diffusion 为最终路线，也不足以宣称扩大模型就能解决这些问题。

## 复跑与资源

```bash
.venv/bin/python -u -m script.train_models --gpu 1 --worker script.benchmark_diffusion -- \
  --dit out/diffusion_benchmark_20261022/release_assets/weights/dit_best.pt \
  --vae checkpoints/vae/vae_final.pt \
  --wave out/diffusion_benchmark_20261022/release_assets/weights/wave_finetuned.pt \
  --run-id diffusion/release-epoch293/seed42 \
  --output out/diffusion_release_recheck \
  --seed 42 --cfg-scale 3 --ddim-steps 50 \
  --min-available-ram-gib 8 --max-rss-gib 4
```

其余生成参数：eta=0、sr=22050、hop=512、mel=128、max_frame=4096、max_window_tokens=256、window_overlap=64，未开启 fixed-length。推理沿用现有 logits>0 解释与 Hold 转换，未添加修复。

此次为 benchmark 增加 `--run-id`，避免外部权重继续被标成“此前本地权重”；报告补充模型结构与实际参数量。23 项适配器/评分器测试通过。8 次推理累计 15.19 秒；进程 RSS 快照 1.574 GiB，torch 显存峰值 allocated/reserved 为 0.594/0.791 GiB，结束时可用磁盘 17.61 GiB。数值来自本机 RTX 3090 单次实验，不是部署整套服务的资源保证。

本地原始预测与日志在 `out/diffusion_benchmark_20261022/release_model/`；四组完整对照为同级 `release_comparison_inputs.json`、`release_comparison_report.json`。[机器可读摘要](diffusion_release_20261022.json)保留来源、评分与产物指纹。权重留在同级 `release_assets/weights/`，VAE 复用已核对哈希的原文件；下载 ZIP 校验和推理完成后清理，权重及预测不提交 Git。
