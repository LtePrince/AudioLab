# R1：固定产物的 Hold 持续处理对照

`script.review_hold_repairs` 对既有手填／自动时基三首 × 4K／free-x 共12份产物做 CPU 重放，不加载权重、不更改产品默认。

```bash
.venv/bin/python -m script.review_hold_repairs \
  --manual-root /path/to/product_smoke_full_v1 \
  --automatic-root /path/to/product_tempo_v1 \
  --output out/hold_repair_v1
```

每个来源目录需含 `song-0`、`song-1`、`song-2` 的完整产品产物。工具检查 manifest 和各文件 SHA、音频长度及 `static-v1` 修复来源；失败保留在12份分母中。输出目录必须不存在。

- A（on）：复用现有修复函数，整个导出字典必须精确等于冻结的最终 JSON，事件也必须等于冻结事件文件。
- B（off）：保留 A 的空间结果，仅将源 Hold 时长恢复为音频末端允许的长度。数量、起点、位置、稳定 ID、floorPosition 和线事件不变；不足1 tick 产生的 Hold→Tap／speed 变化独立记录。
- 两组直接写已有导出模板，不再调用会缩放 floorPosition 的导出转换器。原速度模板必须覆盖 B 的完整尾部。

报告为每个源 Hold 记录原始／A／B 时长、曲末分量、后续起点规则分量与触发音符的 ID、类型、时间和空间位置。触发查询使用相同空间处理之后的位置；同 tick 的行为遵守历史稳定逆序遍历，单独标识，不能称为严格晚于该 Hold 的起点。

`index.html` 汇总12份完整 JSON，并提供一个变化明确的歌曲的 A/B 同步音频预览与片段跳转。预览复用产品 Canvas、原生音频时钟和四键型；共享音频通过软链接引用原产物，需保留原目录。该页面没有触摸判定器，不能证明 B 更可玩，也不能替代听觉／玩家记录。

输出总量限制10 MiB，峰 RSS 限制4 GiB，磁盘和可用 RAM 各保留8 GiB。每份读取前和最终写后检查；最终报告必须真实写入再读回，出现写入、读回或预算错误时删除完成报告，保留失败记录。受控夹具使用模拟预算故障，不实际耗尽资源。

```bash
.venv/bin/python -m pytest -q test/test_hold_repair_review.py
```

测试覆盖同轨／邻近与无触发、空间处理后才进入邻近范围、曲末保留和双因素分量、非零 offset、四种键型、亚 tick 持续转 Tap、同 tick 顺序、floor 不二次缩放、来源损坏与失败分母、真实 JSON 读回和成功标记清理。合成测试与12份开发产物均不属于独立质量评测。
