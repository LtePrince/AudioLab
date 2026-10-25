# AudioLab：自动制谱产品与技术路线讨论交接

整理日期：2026-10-25。

本文件整理最近两轮对话，供另一个会话快速了解用户关注点、查证结果与结论边界。来源链接来自这两轮的公开资料核查，本次整理没有重新开展调研。

## 1. 背景与用户关注点

AudioLab 的目标是从音乐音频生成可玩的音游谱面，包含定轨与无轨路线。这里讨论的是游戏谱面生成；音乐生成、音频转 MIDI、五线谱排版属于相邻任务，不能直接作为同类产品排名。

用户连续提出两个问题：

1. **“市面上音游谱面生成最先进的企业和技术是啥，为啥我都没咋听过”**
2. **“Mapperatorinator不是用的U-net diffusion吗，也算老东西了吧？”**

用户关心的是：这个领域真正领先的对象是谁、技术是否足够先进，以及 AudioLab 应该选择什么对标。第二个问题进一步质疑了把成熟开源项目当成前沿技术代表的依据。

## 2. 第一轮：公开产品与技术调研

### 总体结论

截至本轮调研，没有找到足以支持“某家公司公认世界第一”的统一公开横评。查到的代表对象主要是社区开发者、小团队和研究团队；不同游戏的操作机制、数据集、指标与时间容差不同，不能直接根据论文分数排名。

下列对象属于值得关注或对照的产品／项目，**这不是经过实测确认的领先榜单**。

| 对象 | 公开定位与技术 | 对 AudioLab 的参考意义及证据边界 |
| --- | --- | --- |
| Mapperatorinator／OliBomby | 面向 osu! 四种模式，公开代码与权重，支持音频到谱面事件生成及难度、风格等控制；具体架构见下一节 | 适合做可复现的工程基线。没有据此证明它是当前生成质量第一或架构最前沿 |
| DanceGAI／ManiaGAI | 官方披露为 Beat This 节拍分析、MERT 音乐表示、扩散式段落规划、MaskGIT 式音符生成；支持 4K–9K 及多难度谱集 | 产品工作流与 AudioLab 接近。技术、训练规模、速度和成绩主要来自团队自述，需要独立核验 |
| Beat Sage | Chris Donahue、Abhay Agarwal 开发的 Beat Saber 自动制谱服务 | 较早的研究产品化案例；知名度和历史不能证明当前质量领先 |
| TopMapper | Beat Saber 自动制谱服务，提供多难度、单刀／双刀、灯光与障碍等选项 | 可作为真实体验对照。公开产品页面不足以证明其算法领先 |

来源：

- [Mapperatorinator 官方仓库](https://github.com/OliBomby/Mapperatorinator)
- [ManiaGAI 技术说明](https://dancegai.com/technology)
- [Beat Sage 团队介绍](https://www.patreon.com/beatsage/about)
- [TopMapper 官网](https://topmapper.net/)

### 近期值得跟进的研究

2026-10-10 提交的 **Transfer Learning for Rhythm Game Chart Generation from Music Transcription and Across Games** 研究了预训练音频编码器对制谱的帮助：

- 冻结音乐转录模型 MuScriptor 的编码器，训练投影层与 Transformer 谱面解码器，在该论文的实验设置下优于从零训练。
- 跨游戏／跨模式混合训练不一定带来收益；玩法与取音风格会影响迁移效果。
- 这是值得复现的研究结果，不能直接解释为 AudioLab 已有收益或通用最优方案。

来源：[论文](https://arxiv.org/abs/2610.14955)。

### 为什么这些项目不容易被大众听说

以下是结合资料作出的分析，未进行市场规模或收入调查：

1. **玩法分散。** Beat Saber 的挥刀动作、mania 的键位配置、Phigros 的判定线演出有不同目标，一个系统的效果不容易直接迁移。
2. **高质量制谱包含音乐编排与动作设计。** 对上声音只是部分要求；同一首歌可以有多个合理谱面，接近参考谱不等于手感好、难度自然或有创造力。近期 ChartGenEval 也在研究怎样避免用单一代理指标代替谱面质量。[论文](https://arxiv.org/abs/2607.12857)
3. **社区传播渠道存在限制。** 本轮查到的 osu! Ranked 规则要求音符、音效和 timing 由人直接创作，不接受生成式工具创作。这影响进入排名曲库的渠道；该规则不应扩大解释成不能生成或不能本地游玩。[官方规则](https://osu.ppy.sh/wiki/en/Ranking_criteria#ai-policy)
4. **现有服务主要覆盖缺少人工谱面的长尾需求。** ManiaGAI 官方也承认优秀人工谱师在音乐性和创造力上优于其模型，并把无人制谱的歌曲作为主要使用场景。[产品定位](https://dancegai.com/about)

## 3. 第二轮：Mapperatorinator 架构澄清

用户认为 Mapperatorinator 可能使用 U-Net diffusion，并质疑其技术是否已经陈旧。为回答这个问题，核对了官方主仓库、V32 配置、推理代码和 `osu_diffusion` 模块说明，也查看了本地已有副本。

### 核实结果

| 部分／版本 | 核实到的结构与作用 |
| --- | --- |
| 当前公开主模型 V32 | Whisper 式编码器—自回归解码器，约 2.19 亿参数；输入 mel 频谱，生成时间、音符、位置等谱面事件 token。官方说明包含 RoPE 等修改 |
| `osu-diffusion` 子模块 | 基于 **Diffusion Transformer（DiT）** 修改，生成音符及滑条控制点的坐标。作者明确说明参考了 Scalable Diffusion Models with Transformers，并去掉图像 patchify 和自编码器等部分 |
| V29 的扩散精修 | 主模型生成较粗的坐标后，用扩散进一步细化；当前 README 将这段说明标为 V29 |
| V32 默认推理 | `configs/inference/v32.yaml` 中 `generate_positions: false`，默认不启用扩散坐标生成；V32 增加位置精修 token，官方说明可达 2 像素坐标精度 |

因此，**“Mapperatorinator 当前主模型是 U-Net diffusion”不符合本轮查到的代码与文档**。仓库包含扩散模块，也不意味着每个版本的默认生成路径都使用它。

直接证据：

- [主模型架构](https://github.com/OliBomby/Mapperatorinator#model-architecture)
- [osu-diffusion 架构](https://github.com/OliBomby/Mapperatorinator/tree/main/osu_diffusion#model-architecture)
- [V32 推理配置](https://github.com/OliBomby/Mapperatorinator/blob/main/configs/inference/v32.yaml)
- [推理代码](https://github.com/OliBomby/Mapperatorinator/blob/main/inference.py)：扩散坐标生成受 `args.generate_positions` 等条件控制。

“Whisper 式架构”描述的是网络结构；本轮没有进一步确认其各版本是否使用现成 Whisper 语音预训练权重，后续不要自行把这两件事等同。

### 对“技术老不老”的回答

Whisper 式序列生成、DiT、条件控制都已是成熟技术路线。它们的使用本身不能证明项目具有前沿架构创新，也不能仅凭基础架构的年代判断实际谱面质量。

本轮对 Mapperatorinator 的评价是：其参考价值主要在于完整的谱面表示、数据处理、条件控制与整曲生成实现。**它适合作为可复现、需要实际比较和超越的工程基线，尚不能称为经过横评证明的领域最先进系统。**

助手明确修正了前一轮的表达：应把“值得对标的开源系统”和“经过横评证明最先进的系统”区分清楚；前者不自动支持后者。

## 4. 对 AudioLab 的讨论结论

当前提出的建议是：

- 用 **Mapperatorinator** 对照公开实现、事件表示、条件控制和整曲生成工程。
- 用 **ManiaGAI** 对照音乐理解、段落规划、多难度生成及完整产品工作流。
- 把 **预训练音乐／转录编码器** 列为值得做对照实验的技术方向。
- 用同一批歌曲、明确的玩法和难度条件比较音乐对应、可玩性、可控性、修谱成本与资源使用，而不只比较模型名称、架构年代或参考谱 F1。

以上是研究与对标建议。这两轮没有决定切换 AudioLab 架构，没有完成上述产品的独立实测或盲评，也没有证明任何一个方案已经优于 AudioLab。

## 5. 后续会话接续要点

1. 用户正在追问“真正的技术领先性”，继续回答时应明确区分架构创新、工程成熟度、产品可用性和可验证的生成质量。
2. 讨论 Mapperatorinator 时必须注明模块及版本，保留 V29 扩散精修与 V32 默认配置的区别。
3. 若继续研究，可比较这些方法在音频编码器、谱面表示、段落规划、生成目标、数据规模与评价证据上的具体差别。
4. 若需要评价 AudioLab 当前进展，应重新读取实际仓库和实验记录。本文件只交接上述两轮讨论，不提供当前代码、分支、训练任务或模型成绩的状态快照。
