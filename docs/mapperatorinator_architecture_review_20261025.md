# Mapperatorinator 对 AudioLab 的架构参考

核查日期：2026-10-25。这是代码阅读与后续实验建议，未运行其模型，也未改动正在训练的 E18。

原讨论已原样迁入 [AudioLab_chart_generation_context_2026-10-25.md](AudioLab_chart_generation_context_2026-10-25.md)。原文件8755字节，SHA256为 `52f3b29bf064f2f2498c437b421621faffc73fdf2f2c0bbaac523b0a6b3ab02d`，上级目录原文件已移除。

## 核查版本

- 官方仓库本次读取的main提交：[`0e2b0e387aab4b35c64b0b11b12d47578dea7587`](https://github.com/OliBomby/Mapperatorinator/commit/0e2b0e387aab4b35c64b0b11b12d47578dea7587)，提交日期2026-08-13。以下结论针对该提交的V32配置。
- 本机已有 `Audio/Mapperatorinator` 是 `a8c929c`（2026-03-04），且有用户修改的 `cli_inference.sh`。本次没有更新或改动这个旧副本；V32关键文件从官方固定提交单独读取。
- AudioLab对照版本为 `8a2b235`，当前 E18 / AL-FreeX v0.4.0；4K E14已停在1000轮。E18继续全部300首训练，无歌曲留出集。泛用性数据仍等用户另行提供。
- 原交接文件中的其他产品、论文和市场判断不在本次重新核查范围内。

## 主干相似，差别主要在训练表示与生成组织

| 环节 | Mapperatorinator V32 | AudioLab当前E14/E18 |
| --- | --- | --- |
| 音频 | Mel连续向量，经Whisper式编码器self-attention | Mel经卷积＋双向self-attention，冻结编码器特征作为memory |
| 谱面 | 因果self-attention＋音频cross-attention，自回归事件token | 同样使用因果self-attention＋cross-attention，另有28维因果历史摘要 |
| 时间 | 10ms量化的窗口内时间token | 绝对音频帧pointer＋连续帧内偏移，独立停止头 |
| 坐标 | 粗位置token＋位置细化token；mania有独立列token | 位置与四键型合并NOTE；free-x为64bin＋连续残差回归 |
| 优化 | 移位标签的teacher forcing、逐token CE；V32启用label smoothing | 同样整序列teacher forcing；时间、键型、停止、坐标分项归一化损失 |
| 整曲 | 重叠音频窗口，携带先前生成事件继续 | 整曲音频可见，整曲自回归；另用短波形训练增强 |

证据：[官方模型实现](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/model/modeling_mapperatorinator.py)、[音频/谱面注意力实现](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/model/custom_transformers/modeling_varwhisper.py)、[事件词表](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/tokenizer.py)；本地对应 [transcriber.py](../src/models/transcriber.py)、[chart_decoder.py](../src/models/chart_decoder.py)、[anchored_ar.py](../src/anchored_ar.py)。

音频的序列位置不等于离散词表token：两边的音频memory都是连续特征，谱面输出才是离散事件。我们的音频已经做了self-attention。Mapperatorinator也仍然用teacher forcing训练所有位置，并不是靠一次自主生成一批token再反向传播来解决历史依赖。它说明这一类AR架构有实际工程参照，但不能单凭源码证明其质量优于我们或优于diffusion。

默认训练配置的 `pretrained_path` 为空，模型构造器按backbone config实例化；因此不能把“采用Whisper架构”写成“已经证明使用Whisper语音预训练权重”。也没有核实其发布权重完整训练谱系。[训练默认配置](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/configs/train/default.yaml)

## 最值得借鉴的四件事

### 1. 在历史时间有误时，仍训练恢复正确目标

`data_utils.py`先生成干净的移位监督标签，再扰动 `decoder_input_ids` 中的 `TIME_SHIFT`。V32设 `timing_random_offset: 2`，结合10ms时间步对应逐时间token最多约±20ms扰动；实际范围还受词表边界裁剪。该版本默认第二种共同偏移为0。

这和AudioLab现有增强的目的不同：我们平移波形时同步移动标签，训练的是时间平移响应；这里保留正确音频/目标，污染输入历史，训练的是纠错。它也不同于我们现有的token dropout：丢弃嵌入不等于模拟一个看起来合理但偏了几毫秒的历史起点。[实际扰动顺序与代码](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/dataset/data_utils.py#L606-L680)、[V32配置](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/configs/train/v32.yaml)

建议在E18完成后优先做单因素对照：扰动过去时钟/位置输入，重算对应历史摘要，目标保持干净。要覆盖时间嵌入和摘要等重复携带历史的入口，避免另一路仍提供精确历史。当前pointer有单调可行域，必须先明确“被污染的历史观测”和“输出的合法时间边界”的关系，防止正确目标被噪声移出候选域；不能机械地给所有时间字段同时加噪声。

这是缓解训练/自主生成历史差异的候选方法，不是已经证明能消除误差累积。增删音符、错误键型等生成偏差仍需单独实验。沿用三首整谱、静音/稀疏音频、时间偏移扰动后的恢复、停止率和键型排布评测，不能只看训练CE。

### 2. 将长曲生成边界与音乐内容结束分开处理

官方说明使用约16.4秒音频窗口；推理默认lookback=0.5、lookahead=0.4，中间有效新增部分为10%，配合先前已生成事件和前瞻音频续写。解码器会屏蔽已覆盖区间，进入下一窗口负责的区域时结束当前窗口。**这是明确的窗口调度/解码约束，不能算模型自主学会整曲EOS的证据。** 当前训练配置lookback为0.2、启用概率0.5，也不能把推理比例误写成完全相同的训练配置。[推理默认值](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/config.py#L54-L80)、[窗口处理](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/inference/processor.py)、[时间logit处理](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/inference/logit_processors.py)

对我们的价值是降低长序列停止失败和局部错误连锁影响，同时限制显存和推理开销。用户希望每步能看全曲，因此更合适的待验证方案是“全曲低分辨率memory＋当前区间高分辨率memory＋已有谱面上下文”，而不是直接退回旧版狭窄cross-attention硬窗口。双分辨率memory是我们提出的适配方案，不是该项目已实现的功能。

跨块必须保留活跃Hold状态，并区分块结束、整曲结束和长条延续。上游有Hold开始、结束、sustain事件，值得参考，但不能直接照搬osu!的操作规则。窗口交界要额外测漏音、重复音、长条截断和重复乐句的排布一致性。[Hold事件定义](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/event.py)

### 3. 用明确属性表达编排，不把所有问题交给一个联合分类

上游把时间、对象类型、列/位置、细化、结束/延续等作为不同事件属性，另有玩法、难度、描述标签等前缀条件。我们free-x当前把64个bin和4种键型合为256种NOTE，持续时间和残差另行预测。建议将“键型→位置→残差/时长”的串行分解，或保留单token、改为条件化多头，作为独立实验；两种方式不可混为一项收益。[词表构造](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/tokenizer.py#L145-L224)

这可能让键型学习跨位置共享统计，方便诊断“节奏对而排布错”。但它不自动解决多数类Tap偏置；串行表示还会增加序列长度，比较时需控制训练计算量。先统计键型、位置、时长的独立损失/准确率和梯度量级，再决定表示或损失调整，不能仅凭贪心全Tap就判定历史权重应增大。

难度、操作方式和风格条件对“一首音频有多个合理谱面”有价值，后续数据若有这些可靠标签可加入，并对缺省条件做训练。上游虽有Hold比例条件，AudioLab仍按用户约定以音频和玩法需求决定Hold，不设固定配额。

### 4. 把编码器适配、模型容量和数据规模拆开归因

官方标称完整模型约219M参数；E18可训练部分为6,815,860参数，另有冻结编码器，二者不是同一统计口径。上游直接在模型forward中处理音频并进行谱面CE训练，而我们当前冻结音频特征。我们的输入不是“没做注意力”，但冻结特征是否保留了编排需要的信息，仍值得测。[官方架构说明](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/README.md#model-architecture)、[模型forward](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osuT5/osuT5/model/modeling_mapperatorinator.py#L139-L219)

后续分别比较“冻结编码器/解冻末层”以及“当前容量/适度扩大容量”，使用同一300首和相同训练预算。扩大模型、换表示、加新数据同时发生，就无法归因。上游V32使用web数据、变速/位置翻转、bf16与变长注意力等工程配置；700000步是配置预算，不是发布权重已完成这些步数的核验证明，数据分片索引也不能当作歌曲数量。[V32训练配置](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/configs/train/v32.yaml)

## diffusion、节拍处理与评测边界

`osu_diffusion`是DiT坐标模型；官方将V29的坐标扩散精修单独说明，V32推理配置 `generate_positions: false`。因此它提供的是“时间/对象生成”和“几何排布/精修”可拆分的参考，而不是“当前主干靠U-Net diffusion生成整谱”的证据。[DiT模块](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/osu_diffusion/README.md#model-architecture)、[V32推理配置](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/configs/inference/v32.yaml)

对free-x，等时间和音符类型较稳定后，才值得单独试条件坐标扩散/段落排布生成；它不能凭空修复漏掉的音符、错误起音或整曲停止。对4K，固定四轨的排列是离散问题，不应直接套连续二维坐标扩散。Phigros动态判定线仍是独立的后续任务。

上游默认开启resnap：根据生成的节拍/细分信息调整时间；还有可选的多次timing生成聚合。其最终节奏效果可能同时受神经网络与后处理影响。与AudioLab比较必须分开报告原始输出和处理后输出、相应推理耗时；本项目的网格指标继续仅作匹配加分，不改成强制吸附。[推理调用](https://github.com/OliBomby/Mapperatorinator/blob/0e2b0e387aab4b35c64b0b11b12d47578dea7587/inference.py#L494-L548)

## 建议顺序与本次执行范围

1. 完成E18当前长训，保留A/S独立权重；结合整谱与独立键型指标判断拟合程度。
2. 优先试历史时间扰动；训练/推理条件保持一致，单独记录改变的入口和误差恢复。
3. 若时间较好但Tap/位置坍缩仍在，再做属性分解或损失配比对照；若主要问题是长曲续写与结束，则先做带全曲memory的分块生成。
4. 在上述单因素结论基础上安排编码器适配和容量实验；坐标扩散作为free-x的独立后续方向。

以上均为建议，尚未执行。当前只迁移交接文档、核查源码并记录结论；无新增模型训练、推理规则或采样改动，无外部权重/数据下载。E18运行源码哈希保持不变。
