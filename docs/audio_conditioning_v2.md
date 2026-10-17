# 整曲音频条件审计 v2

`script.audit_audio_conditioning_v2` 对现有两个布局权重执行固定真实 token 历史的特征干预。它修正旧脚本的固定8192帧补齐／截尾、来源与有效尾部问题，增加 free-x 连续 offset 诊断。旧 `script.audit_audio_conditioning` 及历史报告保留；本工具不改变训练、产品生成、采样或修复默认行为。

## 运行与预算

从 `chore/audio-conditioning-audit` worktree 使用原始资源的绝对路径，不调用 `TokenChartDataset`，不写 mel 缓存，不复制音频或权重：

```bash
.venv/bin/python -u -m script.train_models --gpu 1 \
  --worker script.audit_audio_conditioning_v2 -- \
  --data-list /home/ziyaowang/Alvin/Audio/AudioLab/data/val.txt \
  --indices 23 22 24 --split-label reused-development-val \
  --selection-reason 'same three product development examples; not a frozen quality test' \
  --checkpoint-root /home/ziyaowang/Alvin/Audio/AudioLab/checkpoints/timing_v1 \
  --gameplay-dir /home/ziyaowang/Alvin/Audio/AudioLab/data/gameplay \
  --source-root /home/ziyaowang/Alvin/Audio/AudioLab \
  --output out/conditioning_v2 --device cuda:0 --threads 2 --max-rss-gib 4
```

每次必须使用新输出目录。两种布局逐曲串行；3曲循环配对错曲，按音频字节 SHA 验证不同，异路径相同内容也拒绝。首轮固定使用反复使用的开发曲 `いざ参ります_IN`、`sølips_IN`、`ぱぴぷぴぷぴぱ_IN`，不称独立 holdout。时基／offset 来自参考谱；没有加入自动 BPM 因素。

默认磁盘／RAM余量8 GiB，RSS上限4 GiB，GPU fraction 0.65，CPU两线程。原始音频解码沿用产品的预检；full encoder 与 teacher-forcing 在大张量创建前分别预估 L² self-attention、L×T cross-attention／mask、logits、特征与MLP工作集、模型参数。报告记录分项、输入长度、实际可用量与限制；预算不足明确失败，不切换分块或截 token。估算按 float32 数学 attention 的两份分数矩阵计，实际后端可能少得多；它不声称覆盖任意运行时额外开销。

## 特征、参考和评分

复用 `load_audio` 的完整真实音频及 mel：有效 mel 长度为 `ceil(duration / FRAME_SECONDS)`，然后只零补到 encoder stride 2 的最小倍数。memory帧时间为 `i × 2 × 512/22050` 秒，零点保留音频起点；奇数 mel 的最后一个有效 memory 帧保留。每条路线使用自己的 checkpoint 内 encoder，分别记录 raw／padded mel、shape、dtype、内容 hash、有效 mask 和末帧时间。

4K参考读取派生四位置谱，free-x读取既有挖掘 JSONL／sidecar及其原谱，均不称声学真值。相对 `source_path` 只按显式 `--source-root` 解析，源文件缺失或哈希不符明确失败。free-x按线号、上下侧、归一 tick、键型、持续和原始 positionX核对源音符 multiset；相同身份字段的重复项保留 FIFO 关联和歧义计数，不伪称 JSONL 原来具备唯一原生 ID。

每条线原生 BPM 与整谱 offset 参与头尾核验。任何原生／量化后头或 Hold 尾越音频有效域、负持续或负归一 tick 都拒绝整项参考；正 offset 使负 tick 落在音频内也不能绕过当前只编码非负时钟的 codec。保留现有端点量化、位置映射／夹取、同 tick-position 首项去重、短 Hold 转 Tap，逐项记录源ID及变换，编码后再对所有保留事件精确比较 tick／position／type／duration，没有新增修复。

| 条件 | 干预范围 |
| --- | --- |
| matched | 同一曲编码特征 |
| encoded_feature_reverse | 反排有效的已编码特征；不等于倒放波形 |
| wrong_song_resampled_features | 另一曲同权重 encoder 的特征，线性插值到目标有效长度，`align_corners=False`；源索引 `(j+.5)*Lsource/Ltarget-.5`，边缘保持 |
| zero_acoustic_features | 在 `prepare_memory` 前将声学特征置零；投影偏置、memory时间嵌入与真实 token 历史仍存在 |

四条件分别保存干预后特征指纹，以及不变的 tokens／post-token frames／offset targets／offset mask／memory valid mask 指纹。完整真实动作历史可能掩盖音频贡献；这不是端到端错音频生成实验。

`all/time/note/duration` 为下一 token 的参考 bits，保留各自分母与按键型分组的 NOTE token bits。`note_type/position` 在“下一项为 NOTE”条件下分别边缘化类别与位置，不能与 all 相加；DUR token 数也不是 Hold 数。

free-x 连续误差来自同一次前向中消费真实**当前 NOTE**之后的 hidden，使用 `offsets[:-1]` 和当前位置 mask，未错移为下一 token。报告条件亚格 MAE 与乘 `(0.95−0.05)/64` 后的规范屏宽 MAE，以及按键型分母；它以真实 bin 和真实历史为条件，不是完整生成的位置误差。两布局的词表和参考事件不同，不能直接比较其 all bits 大小判断玩法优劣。

## 完成与独立复验

`audit.in_progress.json` 记录运行中进度；`report.json` 只在全部选定行已进入终态后原子写入，再从实际文件读回核验 schema、所选 model/song 唯一分母、来源／特征／tokens等指纹组及shape、特征时轴、固定条件、评分分母、offset单位与有限值。一曲失败保留在分母，终态为 `completed_with_failures`，CLI返回非零；完成报告尾写／读回损坏时删除完成文件，尽力记录 `failure.json`。

报告包括源码 commit／dirty／关键文件 SHA，权重／音频／参考／挖掘源 SHA，设备、精度、线程和种子。每路线单独按 token 或 NOTE 数加权汇总，并保留逐曲差值；差值方向不作为工具通过条件。

CPU用例使用小型真实 encoder／decoder及可控 logits／hidden，验证奇数尾部、超旧190.218秒上限、非零offset／负tick／跨尾Hold、缺源、异路径同音频、全无效memory、资源拒绝发生在编码前、当前NOTE offset对齐、缺失指纹、计数权重及报告写入／读回故障。小权重仅用于接口测试，不作为模型学习效果。

验收按 [Q0–Q7合同](conditioning_audit_review.md) 执行。随后依据真实逐曲结果选择一次单因素生成／修复对照；声学真值、目标播放器与真人评价继续单列，不用此工具替代产品质量验收。
