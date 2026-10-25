# E18：全300首静态 free-x 充分训练

2026-10-25，按用户要求停止4K加训，把训练资源转给静态free-x。**现有300首全部参加训练，不留出验证歌曲**；等本轮训练集拟合充分后，由用户另提供歌曲检查泛用性。E14的4K A/S已完成1000轮，续训断点与重点候选保留；部分早期中间快照后来按用户要求清理，见[清理明细](ar4k_checkpoint_cleanup_20261025.json)。

## 模型身份与训练计划

本轮是 **AL-FreeX v0.4.0**，不是继续使用旧DT解码器训练10轮的v0.3.0。将4K E13/E14的全局音频、因果谱面历史、绝对时间pointer和独立停止头移植到64位置bin＋连续坐标残差。解码器和时间头从零初始化，冻结并复用原free-x音频编码器；不加载4K解码器权重。

| 运行 | 网络的额外起音输入 | 权重与优化器 | GPU |
| --- | --- | --- | --- |
| E18-A / A / `audio` | 两维输入置零 | 独立，seed42初始化 | 0 |
| E18-S / S / `soft` | 连续概率＋帧内偏移，包含弱起音 | 独立，同样seed42初始化 | 1 |

两组共享不可变缓存，**不共享可训练参数、不混合两种条件训练**。A的音频特征仍可能表达起音，“纯音频”只表示没有额外显式起音通道。

每轮300次优化器更新。每首的一次更新联合计算：整曲、两段经过真实波形裁剪和平移后重新编码的片段、同长度静音。全部保留Tap/Drag/Hold/Flick；不设键型配额或双指上限。片段不截断Hold，时间变化不改变横坐标。片段是训练增强，整曲仍能看到全部音频。

沿用4K的学习率历程：先从零训练10轮，AdamW初始3e-4、5% warmup和余弦衰减至3e-5；随后保留权重、AdamW和sampler/torch/CUDA RNG，以3e-5继续到累计1000轮。不是把10轮余弦拉长到1000轮。每20轮进行原benchmark评测并保留紧凑权重；10轮及最后一轮也留档。1000轮是本轮预算，不表示已经收敛或架构上限；后续加训/微调由用户安排。

## free-x必须单独处理的地方

- 4K的时间标记113/114在free-x词表中是合法NOTE。free-x使用保留槽353/354，协议为`audiolab-absolute-time-group-freex-ar-v1`；4K仍使用原协议。
- DT参考序列重编码成时间分组序列时，残差标签按原始`(tick, bin)`重新映射到NOTE位置。平移增强保留原残差，不用bin中心代替连续坐标。
- 输入残差归一化到`[0,1]`；输出屏幕坐标为`x = 0.05 + 0.9 × (bin + residual) / 64`。原Phira静态线坐标换算约定不变。
- 当前NOTE的hidden预测当前NOTE残差。该残差只进入后续token的因果历史；教师强制使用以前参考值，生成使用以前预测值。当前或未来参考残差不会提前输入网络。
- 原时间/键型/停止损失保持：frame CE＋帧内比例MSE＋0.25键型/位置token CE＋0.5事件归一化停止BCE；另加NOTE归一化坐标残差MSE，权重1，沿用旧free-x坐标监督约定。联合更新的各分母独立计算。

## 评测与结论边界

300首整曲均统计教师强制损失、帧定位准确率、键型token CE、停止指标与条件坐标误差。固定Eltaw、SURREALISM、ABYSS进行自主整谱生成，贪心和seed42/top-p0.95分开；记录唯一起点F1（25/50ms）、网格匹配加分、首尾误差、自然结束、完整音符/坐标与四键型分布。网格只作匹配加分，不强制排布。

保留原来的静音、稀疏钢琴和波形平移探针，检查基本条件响应；这些是构造诊断，不是另划一个歌曲测试集。三首固定训练曲的成绩只能说明已见歌曲拟合，不能证明泛用性。连续坐标教师强制误差只衡量给定正确bin/history时的残差拟合；自主生成错误bin和键型需结合保存的整谱另行检查，不能用小残差误差代替完整排布质量。

## 资源、路径与复现

运行根目录：`out/freex_anchored_20261025/`。`pairs/`为free-x专用增强缓存，`pilot/{audio,soft}/`保留10轮完整父断点；`continued1000/{audio,soft}/`为长训输出。滚动恢复用`model_latest.pt`，历史评测用`model_eNNNN.pt`及相邻SHA记录。源码、前端和缓存哈希由配置/manifest记录。原整曲缓存`out/full_corpus_v1/cache`不复制。

启动前资源：两张RTX3090均空闲，可用RAM约38GiB、磁盘约12.8GiB。保留磁盘8GiB、可用RAM8GiB，每个训练进程RSS上限4GiB、显存分配上限65%。额外缓存、两套断点及所有20轮快照预计不超过约3.5GiB；实际资源由保护器持续检查，超预算即停，不自动删除旧权重。

准备缓存：

```bash
.venv/bin/python -m script.train_models --gpu 0 --worker script.train_grounded_ar -- \
  prepare --mode freex --pairs out/freex_anchored_20261025/pairs \
  --output out/freex_anchored_20261025/preparation --max-rss-gib 4
```

每组分别运行，S将`audio`换成`soft`、GPU换成1：

```bash
.venv/bin/python -m script.train_models --gpu 0 --worker script.train_anchored_ar -- \
  train --mode freex --variant audio --epochs 10 --eval-every 20 --snapshot-every 20 \
  --pairs out/freex_anchored_20261025/pairs --output out/freex_anchored_20261025/pilot/audio
.venv/bin/python -m script.train_models --gpu 0 --worker script.train_anchored_ar -- \
  train --mode freex --variant audio --epochs 1000 --eval-every 20 --snapshot-every 20 \
  --continue-from out/freex_anchored_20261025/pilot/audio/model_latest.pt \
  --pairs out/freex_anchored_20261025/pairs --output out/freex_anchored_20261025/continued1000/audio
```

## 启动检查

300首整曲标签已全部通过绝对时间边界与坐标映射检查。606个平移视图也已通过标签检查。最长新协议序列5634 tokens，未截短任何训练歌曲；三首benchmark参考起点与free-x缓存一致。free-x残差映射/因果性/生成反归一化/平移、4K原协议、断点续训与历史回归共45项测试通过。

当前尚无本轮训练质量结论，网页服务默认权重未替换。实时阶段和PID以运行目录中的`supervisor.json`为准，训练轮次以各组`training.json`为准。

2026-10-24 22:39:55 UTC已启动：E18-A/GPU0与E18-S/GPU1，各6,815,860参数、相同初始化SHA，各自独立优化器。后台监督进程检查后续阶段源码不变，在10轮训练及评测正常结束后自动续到1000轮；异常退出会记录原因并保留最近完整轮断点，不自动重试。缓存完成时磁盘剩12.66GiB、可用RAM36.68GiB。
