# AR：借鉴 LLM/VLM 的条件生成训练

本轮针对当前 4K global AR 的条件依赖弱和短音频空谱问题，先保持 tokenizer、全局 attention、四种键型和推理采样不变。所有弱起音仍以连续概率/偏移进入网络；网格匹配仅用于原 benchmark 的加分。本轮不更改产品默认权重。free-x 在共享训练方法验证后另立实验。

## 论文与官方代码对照

| 参考 | 检查到的机制 | 本项目选择 |
| --- | --- | --- |
| [mDPO / CoPO，EMNLP 2024](https://arxiv.org/html/2406.11839v2)，[官方实现](https://github.com/luka-group/mDPO) | 除回答偏好外比较输入模态条件，避免只学语言偏好；另设正例 reward anchor | 首轮借鉴条件排序，并保留监督 CE。采用下面明确列出的轻量损失，不宣称完整复现 mDPO |
| [LRV-Instruction，ICLR 2024](https://arxiv.org/abs/2306.14565)，[官方数据/代码](https://github.com/FuxiaoLiu/LRV-Instruction) | 正负指令均进入训练，研究负例设计及正负平衡 | 借鉴配对正负样本思想；等长波形、时间平移和 token 归一化是本项目自己的适配，并非该论文提出的算法 |
| [BLIP-2，ICML 2023](https://proceedings.mlr.press/v202/li23q.html)，[LAVIS `Blip2Qformer.forward`](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py) | ITC 模态独立编码、ITM 匹配判别、ITG 因果生成，生成 CE 之外还有匹配目标 | 借鉴条件匹配与信息隔离；暂不把整曲逐帧 memory 压成固定 32 个查询，以免损失起点精度 |
| [OpenFlamingo `GatedCrossAttentionBlock` / `MaskedCrossAttention`](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py)，[`FlamingoLayer`](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/flamingo_lm.py) | 跨模态残差乘以零初始化 tanh gate；按图像标记控制交错图文的可见性 | gate 用于保护预训练 LM，当前随机初始化小 decoder 的弱条件路径不宜直接再关掉。图像序号 mask 也不等于局部音频窗。此为 OpenFlamingo 项目实现，不是 DeepMind 原版源码 |
| [Pix2Seq，ICLR 2022](https://arxiv.org/abs/2109.10852)，[`build_response_seq_from_bbox`](https://github.com/google-research/pix2seq/blob/master/tasks/object_detection.py)，[`quantize/dequantize`](https://github.com/google-research/pix2seq/blob/master/utils.py) | 坐标和类别组成输出 token；坐标/类别/结束等监督可分别加权，伪对象有专门标签 | 下一轮独立比较绝对时间锚点与当前累计 DT。保留物理时间分辨率，不直接按全曲长度压成固定少量 bins；不复制对象随机排序或过滤空标注图片的假设 |
| [LLaVA `build_vision_projector`](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py) | 连接器可为线性层或多层 GELU MLP | 当前已有线性 memory 投影。两层 MLP 可作后续廉价单因素实验，不能代替条件训练 |
| [SCST，CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html) | 用自由采样的序列回报训练，以贪心结果作基线 | 基础对齐通过后再考虑；当前只用 onset F1 作奖励会鼓励已观察到的全 Tap 退化 |

普通 scheduled sampling 在错误时间前缀下仍要求原下一 token，标签可能已不一致，因此本轮没有加入。其目标一致性问题亦有[专门分析](https://arxiv.org/abs/1511.05101)。后续纠错训练应重标注合法续写或用序列目标，不能随意破坏时钟后继续套原答案。

## 首轮实现与固定实验协议

共同起点是 `4k-scratch/global/decoder_latest.pt`：6,527,745 参数、已训练 10 轮。三组各继续 10 轮，所有 300 首都参与，学习率 1e-4，seed=42，统一重建 AdamW 并重启相同形式的 warmup/cosine 学习率调度。`data` 与 `rank` 同起点、同样本顺序、同更新数、同正例 dropout；`continued` 保留旧配方的额外静音更新，因此是“旧配方继续训练”参照，不宣称三个实验严格等计算量。

1. `continued`：原整曲 CE，每 10 次音乐更新增加一次短静音 BOS→EOS 更新。
2. `data`：每次更新包含一首原整曲、同曲两个平移短段，以及等长真实零波形。CE 求和后按本次全部监督 token 数统一归一化。
3. `rank`：与 `data` 完全相同，再增加双向时间决策的条件排序损失。

每首原始音频抽一个内容段，放到 6/12/24/40 秒容器中的两个不同时间位置。两条波形重新计算 mel、声学特征和软起音，不截取原全曲 Transformer 特征。容器长度、BPM、有效帧 mask 相同，前后静音是有效输入，不能被 mask 隐藏。所有四型标签同步按物理采样时钟平移，再量化到原有 tick 表示；拒绝裁切 Hold 和时间量化冲突。配对偏移至少差 350ms，并剔除唯一起点在 25ms 容差下重合超过一半的近等价负例。测试钢琴波形及标签不入训。

正例 `(a0,y0)`、`(a1,y1)` 都做监督；在同一个目标 `y` 下，正确/错误波形只交换输入条件，目标前缀及时间不变。只对待预测的 DTB/DTT 计算平均 log 概率；在预测该 DT 之前取分，不向输入提供待预测的新时间。不对轨道或键型偏好施加错误条件惩罚。EOS 保留 CE 监督，不要求所有音乐必须使用同一种结束模式。

本项目的损失为：`L = sum(CE_all_supervised_tokens)/N + 0.2 * mean_both_directions softplus(-2 * (mean_DT_logp_correct - mean_DT_logp_wrong))`。第二项没有冻结 reference、不是全序列 log 概率和，也没有回答偏好项；因此它是借鉴 CoPO 的条件排序辅助损失，不是原论文复现。正确条件 CE 始终保留，报告分别检查正确 NLL、错误 NLL，防止仅靠把负例打坏拉大差距。两次 forward 在相同 RNG 状态下执行以共享 dropout mask，负例不改变后续 RNG 序列。

联合 token 归一化避免单 EOS 独占优化更新，但也可能稀释静音梯度。训练记录保存音乐/短段/静音的 token 数、CE、静音 EOS 损失和有效权重，不能把样本数相等解释为梯度相等。

评测继续使用 benchmark v1 原协议与相同音频：三首既有歌曲、普通/弱音/重复音/跨块钢琴及静音；greedy 与 seed42/top-p=.95 分开评分。另对三首训练曲生成未参与训练的新裁剪/偏移组合，记录正确/错位条件概率差、首音误差和自由生成起点。由于原词表使用相对 DT，整体平移主要改变开头 DT，因此此诊断主要评价首起音和平移响应，不足以证明全曲节奏已改善。

实现位于 `src/conditional_training.py` 与 `script/train_grounded_ar.py`，通过现有 `script.train_models` 资源入口运行。磁盘/RAM 保护线各 8 GiB、RSS 上限 4 GiB、显存比例上限 0.65；每组只保留可恢复的最新 checkpoint，不新增 worktree 或长期分支。结果将在下节记录。

复跑入口如下；默认使用已有全曲缓存、global 初始权重与 benchmark 输入清单，均位于本机 `out/`，不随 Git 分发。`continued`、`data`、`rank` 要使用各自输出目录；已有训练 checkpoint 只能显式 `--resume`，不能悄悄覆盖。

```bash
.venv/bin/python -u -m script.train_models --gpu 0 --worker script.train_grounded_ar -- prepare
.venv/bin/python -u -m script.train_models --gpu 0 --worker script.train_grounded_ar -- train --variant rank --output out/grounded_ar_20261023/rank --epochs 10
.venv/bin/python -u -m script.train_models --gpu 0 --worker script.train_grounded_ar -- evaluate --variant rank --output out/grounded_ar_20261023/rank
```

每组目录保留 `training.json`、`training_resources.json`、`inputs.json`、`report.json`、`diagnostics.json` 与一个可恢复的 `decoder_latest.pt`。资源记录为训练结束时的实测值；训练和评测使用的代码哈希分别记录。准备数据后修正了半 tick 平移可能改变 Hold 时长的问题；606 个已缓存版本全部以修正函数重算并确认 token/时钟完全一致，所以此次缓存有效。

## 执行与结果

三组训练与两种解码的全部 8 例评测均已完成；[机器可读记录](ar_vlm_grounding_20261023.json)保存配置、曲线、来源哈希、分曲指标与平移诊断。以下 F1 均为唯一起点 @50ms，采样为 seed42/top-p=.95。

| 模型 | 三曲采样 F1 宏平均 | Eltaw 采样 F1 | 三曲贪心 F1 | 四个有声钢琴探针 |
| --- | ---: | ---: | ---: | --- |
| 原 global，10 轮 | 0.6037 | 0.5403 | 0.6960 | 已测三音/弱音为空；重复/跨块当时缺测 |
| 旧配方再 10 轮 | 0.5119 | 0.5991 | 0.7214 | 4/4 空谱，F1=0 |
| 配对数据再 10 轮 | 0.5434 | 0.5197 | 0.6960 | 4/4 空谱，F1=0 |
| 配对数据＋条件排序再 10 轮 | 0.4782 | 0.6234 | 0.6954 | 4/4 能出音符，但全部错位，F1=0 |

三组均通过真实静音无音符检查，贪心整曲仍全 Tap。按用户本轮确认，键型上下文和模式持续留到后续专项解决；这里只记录，不把全 Tap 设为时间定位阶段的阻塞条件。时间任务本身仍未通过：排序组的三音探针贪心输出 9 个音符、采样 4 个，均没有匹配到参考起点。

6 个新平移短音乐版本的结果更能区分训练收益：旧配方全为空谱；配对数据组全部出谱，但三对音频平移后首音的变化量都为 **0 秒**。排序组开始跟随平移，三对参考位移为 0.750/1.121/1.321 秒，生成首音位移为 1.740/1.501/1.554 秒，仍有明显误差。六个首音的平均绝对误差从配对数据组的 594ms 降至排序组的 373ms。

排序组的新变换诊断中，正确条件 DT NLL 为 0.878，错误条件为 7.385；数据组为 0.889/0.891。条件差明显增大主要来自错误条件概率降低，并没有转化成足够准确的自由定位。原整曲采样的宏平均下降，单独 Eltaw 上升，不能据此更换产品模型，也不能声称 VLM 方法已经解决音频对应。

21 项相关测试通过，包括四键型时间平移、Hold 边界/时长、配对差异、DT 监督位置、共享 dropout RNG 与既有全局 attention/缓存运行测试。数据缓存约 138MiB；三组训练分别用时 429/960/1301 秒，单进程 torch 显存分配峰约 0.25GiB、训练 RSS 约 1.62–1.75GiB，最终剩余磁盘约 16.75GiB。此次保留实验代码与证据，未替换产品权重。

## 下一轮结构候选：时间定位与组内排布

以下仅为代码对照后的设计，尚未实现或验证。Pix2Seq 的坐标/类别序列提供表示参考；可变长度输入位置的选择另参考 [Pointer Networks](https://arxiv.org/abs/1506.03134) 与 [Kool 等作者官方 attention model](https://github.com/wouterkool/attention-learn-to-route/blob/master/nets/attention_model.py) 的 `_precompute`、`_get_log_p`、`_one_to_many_logits`。后者是路径规划实现，不能复制“访问过的位置永久不可选”的约束。

最小可检验协议：`BOS → 预测下一时间或结束 → 该起点下生成 NOTE/DUR → END_GROUP → 下一时间或结束`。时间头使用历史状态查询全部音频帧，原 AR 生成同起点的多音符、类型、位置与 Hold 时长。模型预测时间，弱起音作为连续证据；不是外部阈值选起点后强制填键型。

- 在尚未消费目标 DT、下一音符或下一时间 embedding 的状态上监督时间，避免把答案提前喂给 query。训练可在定位损失之后用真值锚点监督排布，评测必须用预测锚点自由生成。
- 增加版本化组结束符或独立组结束头；Hold 的 DUR 串完整结束后才能结束组。同一时刻和弦共享一次锚点，不增加手指数上限。BOS 直接进入定位/停止头，避免继续由旧词表的 EOS 概率主导空谱。
- 当前 memory 间隔约 46.44ms，帧索引还需帧内偏移。两个不同起点可以处于同一帧，不可简单屏蔽上一帧及之前所有帧；最终连续时间与历史、导出保持同一约定。若继续喂原 DT 作为历史，必须处理其 tick 量化损失。
- 停止头与位置 softmax 分开，避免音频越长、位置类别越多而直接改变 EOS 的竞争关系；监督真实静音和末组结束，明确记录停止/继续的权重。
- 不能直接复用当前 `skeleton_ticks` 推理分支：该分支拒绝历史模式，强制 DT 循环也没有同步传入历史状态。新增实现需验证并行训练与缓存逐步推理一致。

先验收三音、同帧连续音、同起点和弦、弱音、静音与提前结束，再跑原整谱 benchmark。该改动主要针对时间与停止，不宣称它自动解决全 Tap、楼梯或重复/模进；这些仍须独立训练与评估。
