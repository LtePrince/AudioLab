# E23-A：完整音频编码器与分窗解码器从零训练

用户要求重新从随机初始化开始。本实验为 **AL-AR4K v0.7.0 / A / E23-A**，保持 E22 的 1337 万参数结构，但音频编码器、谱面解码器、优化器与随机采样状态全部重新初始化。不是从 A1400、分窗 e100 或联合微调 e20 续训。S 继续暂停。

## 初始化与数据边界

`--initialization scratch` 分支只读取[结构 JSON](assets/joint_audio_scratch_20261102/architecture.json)，不调用 `torch.load` 读取任何父权重；配置中 `parent`、`frontend`、`parent_epoch` 均为空，`inherited_parameter_tensors=0`。随机种子 42；采用各模块标准初始化，LayerNorm 的一/零、窗口及历史投影的默认零初始化属于结构定义，不来自旧模型。

卷积层、六层音频 self-attention、memory 投影、全部谱面 self/cross-attention 与时间/键型/位置/结束头都可训练。模型仍随时读取整曲音频，只输出当前 10 秒窗口的音符。纯 A 没有独立起音网络，不读取冻结神经特征作为输入。

只读复用 `out/joint_audio_a_20261102/mel`：全部 300 首的固定 log-Mel、两份原物理裁切、真实静音和原标签，缓存总计 1.438 GiB。共享目录必须已经完整且通过 SHA 校验，不再复制；每一步从 Mel 重新计算当前网络特征。

## 训练安排

- 目标 300 轮、90,000 次更新；不使用微调的 `3e-6` 小学习率。编码器和解码器均以 `3e-4` 为目标学习率，前 1500 步线性 warmup，随后保持恒定；每组都是正学习率。
- AdamW 初始状态为空。每轮保存完整模型、优化器和 RNG，仅保留 latest 与 best-fit。300 轮是首个观察预算，是否充分由拟合与自由生成曲线判断。
- e0 只测 Eltaw 与两首缺音曲，记录随机起点；每 20 轮测同三首，第 100/200/300 轮测固定 15 首训练曲＋10 首社区开发曲。新歌仍为开发回归集，不冒充盲测。
- 指标保持起点 F1、网格加分、时间＋类型 F1、训练集位置联合分、窗口结束、段落缺音与 Hold 覆盖。四种键型完整保留，网格仍只作评价，不强制吸附。
- 继续采用 RSS 6 GiB、RAM/磁盘至少保留 8 GiB、GPU 65% 分配上限与激活重算。网络容量、窗口协议、数据及损失不变；学习率和初始化改变，所以本轮回答从零训练能力，不是严格的单变量冻结消融。

## 启动验证

51 项测试通过，含禁止初始化读取权重文件、相同 seed 可复现、不同 seed 同时改变音频/谱面参数、空 AdamW 状态、warmup 在恢复时按绝对步数计算，以及旧 4K/free-x 的兼容检查。

真实 GPU 检查覆盖 Eltaw、最长音频 Drop It、最长谱面「祈 -我ら神祖と共に歩む者なり-」。全部 279 个参数张量都有有限梯度且实际变化，各音频层均有非零梯度。探针更新全部丢弃，正式训练重新创建同一 seed 的初始网络。资源峰值约 RSS 1.92 GiB、GPU 已分配 0.371 GiB；不是所有歌曲或服务端资源认证。

独立后台任务已启动，先运行三首随机基线，再自动训练。完整配置见[配置与初始化指纹](assets/joint_audio_scratch_20261102/config.json)，梯度见[真实整曲证据](assets/joint_audio_scratch_20261102/gradient_probe.json)。实时状态：`out/joint_audio_scratch_20261102/training.json`，日志：`worker.log`；此时尚无训练后质量结论。

上一组 E22 联合微调已完整结束 20 轮，权重和报告作为对照保留，见[E22 最终结果](joint_audio_a_20261102.md)。不替换已发布模型包。

## 第177轮曲线快照

用户请求训练曲线，已冻结到完整 e177：总损失 4.4681→0.1070，训练帧内条件 MAE 约 2.95 ms；这是给定正确历史和正确时间帧的指标，不是自由生成起点误差。图使用每轮原始值，没有平滑。

![E23-A 训练与固定三首生成曲线](assets/joint_audio_scratch_20261102/curves_e0177/training_curves.png)

[PDF](assets/joint_audio_scratch_20261102/curves_e0177/training_curves.pdf) · [训练 CSV](assets/joint_audio_scratch_20261102/curves_e0177/training.csv) · [生成评测 CSV](assets/joint_audio_scratch_20261102/curves_e0177/evaluation.csv) · [快照与来源](assets/joint_audio_scratch_20261102/curves_e0177/provenance.json)。

下排始终是 Eltaw、天使、匙三首独立曲线，不把第100轮的15首训练曲/10首开发曲均值混入。最新生成检查为 e160：Eltaw F1@50ms 0.9129、网格0.7010并正常结束；天使0.5089/0.1578、仍有3个强制窗口；匙0.6710/0.5319、仍有10个强制窗口，最长参考有音空档约7.60秒。F1及网格保留了结束失败的输出，不能代替成功率或可玩性。这是阶段快照，300轮训练仍按原计划继续。
