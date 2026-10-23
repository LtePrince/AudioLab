# AR 续训：先检验既有结构的训练充分程度

2026-10-23。用户要求保留[定长归因/编谱计划](../fixed_length_attribution_plan.md)，先继续训练未充分训练的模型，允许过拟合，不能再次用短跑结果替代架构能力判断。新定长课程和 diffusion 结构实验暂不执行。

## 运行状态

本记录初始状态为 **running**，尚无长训结论。训练脚本 commit `37a3650`。原 epoch-10 的 `out/anchored_ar_20261023/{audio,soft}/model_latest.pt` 只读保留；新结果在 `out/ar_saturation_20261023/{audio,soft}`。

两组各从累计 epoch 10 / step 3000 继续：

- `audio`：连续声学特征，显式软起音置零。
- `soft`：相同声学特征，保留全部软起音概率/偏移。

保持 6,626,691 参数、全量 300 首、两份真实平移短段/等长静音、原损失、dropout、全局 cross-attention、解码/采样不变。恢复 AdamW 动量、Python 样本采样 RNG、torch CPU/CUDA RNG。续训保留实际末步学习率 `3e-5`，不重置 warmup；原 `3e-4` 仍记为 pilot 基础 LR，实际续训 LR 单列。编码器仍冻结，结论范围是当前前端/数据/损失条件下解码器的拟合能力，不是理论上限。

完整目标暂记累计 300 轮，首个观察段到 100 轮；到达观察点后根据曲线决定继续，不以轮数本身宣告收敛。每 20 轮运行同一三首训练曲、四个正例钢琴探针、真实静音与平移诊断，分别记录 greedy 和 seed42/top-p=.95。评测在 RNG fork 内执行，不消费训练 dropout 的随机状态。固定 benchmark 不入训。

## 可恢复性与资源

每轮原子覆盖 `model_latest.pt`（含 AdamW/RNG），仅保留一个额外 `model_best_fit.pt`（无优化器，只作推理）；评测保存小型结果到 `evaluation-eNNNN/`。最佳拟合依据汇总训练目标，不能等同于最佳编谱。原基线不覆盖，缓存不复制，不新增长期分支/worktree。

限制：剩余磁盘至少 8 GiB、可用 RAM 至少 8 GiB、单进程 RSS 上限 4 GiB、单 GPU 分配比例 0.65。两张 3090 分别运行两组。启动前剩余磁盘约 16.6 GiB、RAM 约 37 GiB。

续训入口检查 schema、变体、网络参数、数据缓存哈希、损失/解码策略、完整 epoch、优化器/RNG。与 pilot 训练提交 `099ee90` 的 AST 比较确认 `encode_target`、`time_intervals`、`AnchoredAR`、`normalized_loss` 相同；先前仅推理边界/失败诊断发生修正。27 项测试通过，包括 AdamW/RNG 恢复后下一次更新逐张量一致、评测 RNG 隔离以及原模型测试。

## 判读规则

- 训练集对应得分在本轮作为可拟合性诊断，不能升级为复杂编谱的唯一目标。自然曲参考谱仍是多个有效编排之一。
- 同时看训练定位/排布/停止损失、完整自由生成、末段覆盖、四键型/模式和探针错漏。真实历史指标不冒充自由生成指标。
- 整曲尾部继续生成仍按失败记账；失败前事件只作局部诊断，不能把这种输出的 F1 当完整通过。
- 持续下降意味着仍可训练；平台需记录观察窗口与优化条件。预算到期不能标“已收敛”。不以独立歌曲泛化损失早停，不因过拟合主动停止。
- 不宣称任何有限训练实验穷尽架构理论上限；如果训练目标拟合而自由生成仍差，需区分训练/推理差异与结构能力。

## 命令

两组分别指定 `--gpu 0/1`、`--variant audio/soft`：

```bash
.venv/bin/python -u -m script.train_models --gpu 0 \
  --worker script.train_anchored_ar -- train --variant audio \
  --continue-from out/anchored_ar_20261023/audio/model_latest.pt \
  --output out/ar_saturation_20261023/audio \
  --epochs 300 --until-epoch 100 --eval-every 20
```

恢复同一运行加 `--resume`，保持其余配置；延长观察段只改 `--until-epoch`，不会重置恒定学习率。最终结论与曲线在运行获得实测后补充。

## 第 20 轮观察（仍在续训，未收敛）

| 指标 | epoch 10 audio / soft | epoch 20 audio / soft |
| --- | --- | --- |
| 训练定位 CE | 1.18973 / 1.18978 | 1.07968 / 1.07987 |
| 训练帧准确率 | 62.041% / 62.052% | 65.327% / 65.311% |
| 训练末尾停止召回 | 30.75% / 30.75% | 35.167% / 35.0% |
| 三首整曲自然停止，greedy 和 seed42 分别 | 两组均 0/3、0/3 | 两组均 0/3、0/3 |

两组训练曲线继续改善，还不能认为训练充分。整曲错误仍是 `continued with no future audio interval`，所有完整生成仍失败。Eltaw seed42 的失败前局部起点 F1 为 audio 0.68005、soft 0.67957；这仅是训练集部分输出的定位诊断，不能作为完整成谱质量提升。

普通三音 seed42 两组都只生成约 4.006 秒处的一个双 Tap（一个唯一起点），F1=0.5，只匹配 1/3 真值且漏掉另外两次；不能将它称为通过。greedy 仍产生 35 个起点且尾部停止失败。弱音 seed42 F1=0.14815，跨块 seed42 F1=0.04878；纯静音为空谱。

greedy 的三首自然曲仍全 Tap；Eltaw seed42 的 audio 为 Tap 782 / Drag 91，soft 为 Tap 774 / Drag 91。没有按类型数量选模。平移诊断中 SURREALISM 首音约 1.103 / 1.935 秒（真值 1.198 / 1.948 秒），Eltaw 的两次首音仍约 3.042 / 2.571 秒，顺序问题仍在。

原始结果：`out/ar_saturation_20261023/{audio,soft}/evaluation-e0020/{inputs,report,diagnostics}.json`。评测后单进程 RSS 约 2.1 GiB，torch 显存峰分配 0.306 GiB / 预留 0.531 GiB，剩余磁盘约 16.36 GiB。训练自动进入后续 epoch，未停止或切换默认产品权重。
