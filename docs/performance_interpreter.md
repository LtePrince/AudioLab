# 原始动态谱查询与重建审计（F4a）

本模块属于 `feature/freeform-gameplay`，读取原始 ID JSON，保留每条线的独立时钟、曲线与音符身份。它为后续演出标签提供可查询表示；没有新增演出权重，也不将静态 `generate.json` 当动态监督。旧 `gameplay_miner.py` 与当前生成产品的行为不变。

## 输入和规范文件

`PerformanceChart.from_file(original_json, audio_path, split=...)` 只读原谱和音频头，记录源文件／音频 SHA-256、来源路径、划分和音频长度。支持 PGR v1/v3。规范文档保存完整源 JSON、原始字段与未知扩展字段，同时保存已解码的逐线事件和音符。源 line/note/event ID 根据源列表索引形成，不排序重写源文件，不将局部 line ID 当跨曲语义类别。

每线按 `audio_s = native_tick × 60/(32×line_bpm) + offset` 换算。保留浮点 native tick，不统一整数化；音频秒转回 native 时只在源边界附近恢复不超过若干 float64 ULP 的舍入误差。该操作没有节拍量化。v1 打包坐标使用截向零的商与带符号余数，`-1001` 表示原像素 `(-1,-1)`，不使用 Python 的正余数。

规范文件采用标准 gzip JSON，保留三种独立指纹：原始文件 SHA、规范 JSON 内容哈希、实际压缩文件 SHA。审计先写文件，再用 `gzip.open`／`json.load` 读回并建立第二个查询对象，核验字段和完整查询输出；这证明表示往返一致，不能代替独立数值真值。

## 显式解释策略

`src/data/performance.py` 的 `native-clock-right-continuous-segments-v1` 是声明过的数学解释，不宣称官方渲染器语义。

| 情况 | 处理 |
| --- | --- |
| 正时长事件 | 在原生浮点时间插值，区间右开；共享端点归右侧开始的事件 |
| 零时长事件 | 只在该瞬间提供 end 值；同刻存在右侧开始的正时长事件时，后者优先；冲突零段标 unsupported |
| 外部时间 | 在最外端点之外保持端点值，报告 `outer_endpoint_hold` |
| 内部间隙／重叠 | 查询返回显式 unsupported；不借最后端点或插桥填补 |
| 反向区间／非法事件 | 原字段保留；反向覆盖域或非法轨道明确不支持，不静默删除 |
| 源事件顺序不齐 | 保留源顺序和 ID，查询索引单独排序，报告顺序诊断 |
| 旋转 | 保留连续角度和转数，不包到 ±180°；三角投影才使用其周期性 |
| 显隐 | 保留原 alpha；另提供 `max(0, alpha)` 掩码，不将大于1的源值裁成1 |
| 速度 | 原值保留，负速允许；分段常速按源线秒积分，原点为 native tick 0；历史缺口使累计值未知 |

区间的左／右极限通过 `interval_values()` 取得，不能把右连续的 `query(b)` 当作上一段的末值。这个区别用于显隐整段证明及旋转采样数；跳变单独记录。测试覆盖末端短时 720° 旋转再跳回 0、短正 alpha 区间再跳负值等反例。

`line_state()` 分别返回 `current_state_supported`、`scroll_supported`、总 `supported` 及 `within_audio`。请求积分时，总支持包含累计值有效性；`effective_supported` 还要求位于音频域。原事件／音符问题与其是否和有效音频域相交分别统计，完整位于音频之前／之后的源问题不会计入播放域内分母。

## Hold 与空线

16:9 规范视口以左下为 `(0,0)`。判定几何为 `x=Lx+0.05625·p·cosθ`、`y=Ly+0.1·p·sinθ`，区分屏宽与屏高单位。`sample_hold()` 保留整段源 Hold，采样包括全部曲线边界、单侧检查点和内部点；默认连续区间内不超过0.25秒及22.5°旋转间隔，累计滚动也默认查询。超过2万点的单 Hold 明确拒绝，不静默截尾。跳变不属于连续角度间隔保证。

报告分别计数完整源时段与音频内样点的未知状态／越界。采样包含内部状态，不构成连续安全、视觉可读性或触摸跟随要求的证明；这里的世界位置是判定几何目标。测试含头尾与中点均在屏内、四分之一处越界的整圈 Hold。

有音符与音频域相交的线标为 gameplay，同时保留源／有效音符数。其他线保留为 `visible_decoration_candidate`、`idle`、`sampled_not_visible` 或 `unknown`。只有完整支持的分段 alpha 极值均不正等可证明条件才判 idle；有限样点都不可见不能证明全程闲置。潜在可见性使用无限线与规范视口相交，不推断源 JSON 没有给出的纹理长度、厚度或遮挡。

## 固定播放器源码参考

参考注册在 [phira_reference.json](phira_reference.json)，固定 Phira commit `a22ea8958410ac940adec83b54bbf4e6b293d191`。`performance_reference.py` 是根据 PGR／Anim／Tween 源码写的独立 float64 诊断适配器，不执行原播放器，不复现 Rust f32 的每步舍入，不生产训练标签。重复 key 的精确时刻可能受 cursor 历史影响；时间不递增或此类重复时刻明确返回参考 unsupported。

与数学源曲线分开记录的已知参考行为包括：首 key 之前可线性外推、末 key 后保持；曲线转换弹出最后一个 key；速度首事件 start 改0；滚动积分终点为各线最大音符头时间的最大值加1秒，可能早于最后一个长 Hold 的尾；Hold 的 note.speed 设为1，原 floorPosition 不作计算输入。测试包括单个非哨兵末事件、最后长 Hold、全部非空负头与包含空线的 max-head 差异。移植差分与规范往返各有分母，不将自洽或差分接近当作官方兼容认证。

## CPU 审计

```bash
python -m script.audit_performance \
  --data-list /path/AudioLab/data/train.txt \
  --output out/performance_v1 --reference-cache /tmp
```

可选 `--reference-cache` 检查缓存中 `audiolab-phira-{pgr,anim,object,tween,core}.rs` 与注册 SHA 一致；源码地址均使用固定 commit。脚本按训练表顺序选择两份 v1 和四份 v3，只用于工程覆盖，不是质量总体抽样。它查询整个音频域的源边界及内部点，检查所有源 Hold，分别报告格式解析失败、未知区间、参考不支持、样点差异和实际序列化往返。

运行采用 CPU，RSS上限4 GiB，可用RAM／磁盘保留8 GiB，总产物预算50 MiB。没有训练、权重复制、音频复制或全库高频缓存。最初未压缩草稿的第六份规范输出触发45 MiB子预算，被明确记为失败；其摘要保留，生成的未压缩草稿副本清理后改用 gzip 进行六份完整重验。这次资源失败不会从草稿分母中消失。
