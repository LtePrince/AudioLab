# 已有整谱的 pattern／几何诊断

`script.diagnose_patterns` 只读已有三曲×手填／自动时基×两玩法的12份完成产物，分别描述 raw 与 final，并复用 C2 的同曲参考和音频条件结果。它不加载权重、不推理、不改生成默认、不写新的谱面。输入是反复使用的开发样例，不是冻结质量测试集。

运行需原始生成目录和 C2 引用的参考路径仍可读；脚本不会下载或复制数据，也不建立 Dataset cache：

```bash
python -m script.diagnose_patterns \
  --manual-root ../AudioLab-product-development/out/product_smoke_full_v1 \
  --automatic-root ../AudioLab-product-development/out/product_tempo_v1 \
  --conditioning-report ../AudioLab-conditioning-audit/out/conditioning_v2/report.json \
  --conditioning-sha256 26de0aa70abef8772ee45f15c251cf929275aa34ce7b095a53a21c38b1092114 \
  --output out/pattern_diagnostics_v1
```

输出目录必须不存在。产物上限10 MiB、进程 RSS 上限4 GiB、磁盘和可用 RAM 余量各8 GiB，沿用已有 CPU 重放保护。各行失败保留错误与12行分母；最终 JSON 写入／读回／资源检查失败会删除完成文件。资源测试使用 mock。

## 描述器定义

- 每条谱使用自己的整数 tick 分组，同刻多个音符作为一个 onset 组。时间先用 `tick × 60/(32×BPM) + offset` 转为音频秒；编码 BPM 不视为音乐真拍。raw/final 用稳定 ID 一一对应，改变数量、ID 或头时间会拒绝比较。
- 固定取全局连续四个 onset 组，只有四组都是 singleton 才分类，不能跳过和弦。lane/bin 全同为 `AAAA`；`A,B,A,B` 且 `A≠B` 为 `ABAB`；位置严格递增或递减为 `monotonic_run`；其余 `unclassified`。允许窗口重叠，报告全部窗口、被和弦打断的窗口和各类数量。此处没有等拍筛选。
- 三个相邻音频时间差为 IOI；`ioi_cv = population_std(IOI) / mean(IOI)`。空间类别与节律分开描述。
- free-x 使用原有64 bin、区间 `[.05,.95]`。AAAA 复现偏差为 `max(x)-min(x)`；ABAB 为 `max(|x0-x2|,|x1-x3|)`。单调序列列三个有符号步长、绝对步长的总体标准差和 CV。单位是屏宽比例；修复后同点使均值为零时 CV 为 `null`。
- 同刻组合保留押数、排序后的 gap 和 span；至少三个点时列 gap 的总体标准差。它描述等距程度，不认定等距为设计目标，不设多指上限。
- 四个 singleton 的精确 `(lane/bin,type)` 序列可统计复现次数；该计数允许重叠且不包括节律，不叫乐句识别。4K 与 free-x 的词表、音符和窗口分母不同，不能按复现率直接排名。
- Hold 邻近起点仅记录头尾严格内部、距离小于 `.04` 屏宽的音符，含其键型与 ID。此工程距离取自既有 free-x 规则；计数不是物理碰撞、最少手指或错误判定。曲末／未来起点裁剪的独立归因沿用 [R1](evaluations/hold_repair_v1.md)。

所有分布同时给 `count/mean/median/min/max`，空分母给 `null`，不写 NaN。完整计数使用全部事件，示例最多每类3个、和弦12个、Hold 8个；修复逐音差异完整保留，原始 pattern 的前后配对固定 raw ID，避免重新挑选实例。

## 候选与来源

原始／最终谱、事件文件、修复记录、候选、音频由已有 manifest 哈希验证；候选用当前未改行为的 `quantize_candidates` 复现合并／边界钳制，并与冻结报告核对。列出缺失／新增 tick、每候选音符数分布；覆盖不等于声学准确率。

若存在 `audio_events.npz`，核对其 SHA，只读取 `event_probability`、`event_times_seconds`、`valid_event_time` 三个小数组，每成员上限1 MiB；不会加载 features。重放原阈值和 NMS，要求选中 ID／时间／分数与已保存候选完全一致。列有效帧、阈值前后和 NMS／重复时间去除分母。模型分数仍未校准，不能据低分推断弱拍、乐句弱起或声音弱能量。

C2 必须匹配调用者指定 SHA，并通过已有 validator；各产品按音频 SHA 和模型 SHA 链接 C2。参考重新使用 C2 的完整 codec，要求全部元数据／输入哈希／token 指纹一致；源参考音符数与 tick-bin 去重后数分开。参考是已有人工编排的静态投影，不是声学真值或唯一正确答案。

## residual 中心反事实与音频条件

free-x 的可选解释列保持头、持续、键型、bin 和 ID，仅把 residual 改为 `.5` 后重算几何。单音移动数学上限为 `.9/64/2 = .00703125` 屏宽；它没有重新采样，也没有作为历史反馈。基于 bin 模板定义的复现偏差因此减小是预期结果，不是质量改善。本入口不导出该反事实，不改变模型或产品。

报告保留 C2 两路线全部加权分项，包括方向相反的持续、位置与 offset 结果；它们来自真实参考历史下的 teacher forcing，不是这次生成产物的条件消融。O1 后才根据证据选择一个有限因素；该工具本身不启动 rerank、吸附或训练。

验证：`python -m pytest -q test/test_pattern_diagnostics.py`。这是数学、来源和失败行为测试，不替代 Phira 真机导入、触摸与音乐判断。
