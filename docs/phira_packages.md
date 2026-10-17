# Phira 首版谱包交付

目标是手机／平板触屏上的 Phira，首版提供四个固定触屏位置的4K和静态free-x；保留 Tap、Drag、Hold、Flick 与多指。此轮不改生成权重、取音、Hold修复或动态演出。

## 格式依据

固定 [Phira v0.8.2](https://github.com/TeamFlos/phira/releases/tag/v0.8.2)，提交 `c830dee5e07a2b8f06104576f30d8045e0859aa7`。源码和官方文档的文件 SHA／大小见 [引用清单](phira_package_reference.json)。

[官方谱包说明](https://teamflos.github.io/phira-docs/chart-standard/index.html)要求压缩包根目录直接包含 `info.yml` 及其引用的文件；[元数据说明](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)定义谱、音频与封面字段。[固定导入代码](https://github.com/TeamFlos/phira/blob/c830dee5e07a2b8f06104576f30d8045e0859aa7/phira/src/scene.rs)解压后加载并修正信息；实际游玩另经 `GameScene::load_chart` 与媒体解码。本实现核对这些源码条件，没有运行 Phira 二进制或手机实测。

每种玩法一个 ZIP，根目录5个普通文件：`info.yml`、`chart.json`、`audio.wav`或`audio.ogg`、`illustration.png`、`audiolab.json`。谱与音频保持已验证产物的原字节，封面为自有几何默认图。`info.yml`使用 JSON 流式表示的合法 YAML；明确引用 `chart.json`，因此额外的溯源JSON不会参与文件猜测。`format`留给播放器推断；源码通过原文本中的 `"META"` 判断RPE，本profile拒绝该标记、BOM及非PGR静态模板。

`difficulty=0`仅为格式占位，`level`显示“未定级 · 4K/free-x”，报告为 `unrated_placeholder`，不是难度预测或训练条件。音乐作者标“未提供”，谱师标模型生成；`source.kind=synthetic_calibration`时明确改为合成校准，不能混为模型结果。

## 时间与触屏语义

[PGR解析器](https://github.com/TeamFlos/phira/blob/c830dee5e07a2b8f06104576f30d8045e0859aa7/prpr/src/parse/pgr.rs)把每线 tick 转为 `tick*60/(32*bpm)`，保留谱面offset。[游戏时钟](https://github.com/TeamFlos/phira/blob/c830dee5e07a2b8f06104576f30d8045e0859aa7/prpr/src/scene/game.rs)从音频时间减去谱面、元数据和设备offset的合计。因此本包保留 `chart.offset`，`info.offset=0`；正谱面offset对应延后击打。官方根格式文档对正负方向的措辞与此源码不同，本实现以固定版本执行链为准。设备offset仍由使用者校准。

音频保持原采样率和字节；分析使用22050Hz重采样后的向上取整长度，两者时长最多允许相差一个分析样本（约45.35微秒），报告记录原时长、分析时长与差值。不会重编码或移动原音频。

[触屏判定源码](https://github.com/TeamFlos/phira/blob/c830dee5e07a2b8f06104576f30d8045e0859aa7/prpr/src/judge.rs)分别处理新接触的Tap／Hold头、滑动触发的Flick和接触覆盖的Drag；Hold后续检查附近接触并允许源码定义的短暂离开容差。它遍历触点，没有在本产品中新增双指上限。四固定位置只是谱面布局，并非给Phira增加传统四键输入模式。源码核对不能替代设备延迟、多指识别和实际手感检查。

当前导出采用覆盖所有头尾的恒正速首段及零速末哨兵。本包严格验证这一窄模板，拒绝额外、重叠、乱序变速段。Phira速度积分末端使用最大头时间加1秒，但不能仅以Hold尾越过此时间就断言当前两段模板有视觉截短；本实现不为此增添假音符、裁短Hold或修改线事件。其他动态速度表示不在本次兼容范围。

## 入口与资源

新 `generate_product` 作业在两条布局完成后打包，成功标记之前完成ZIP CRC、各成员哈希、实际读回、PNG与完整音频流解码检查。任何打包失败／取消均删除本次可下载ZIP且不留下完成manifest。网页结果首要动作是每模式完整ZIP下载；预览随模式切换下载目标，原JSON保留辅助。下载ZIP传到设备，在Phira本地谱面中导入；本地服务仍只绑定loopback，无LAN部署。

为已有完成任务补包，不加载权重：

```bash
.venv/bin/python -m script.package_phira \
  --job /path/to/completed-job --output out/phira-package
```

必须使用新的独立输出目录；不改源manifest或源预览。目录包含每模式ZIP、下载页和 `packages.json`。包内不引用原工作区，原谱／音频／封面都包含在ZIP内。API为 `build_phira_package(chart_path, audio_path, output_path, mode=..., title=..., duration_seconds=..., source=..., check=...)`；`check`供取消检查。

音频≤128MiB、谱JSON≤8MiB、单ZIP≤160MiB；解码前限制600秒、8声道、192kHz，每次最多65536帧；磁盘／RAM保留8GiB，RSS≤4GiB。先预估所需磁盘，再流式复制与校验；不把完整音频读入内存。ZIP安装使用不覆盖已有文件的原子操作；失败清理本次临时文件。测试模拟资源拒绝，不消耗真实预算。

```bash
.venv/bin/python -m pytest -q test/test_phira_package.py test/test_product_pipeline.py test/test_product_entry.py
```

包内校验通过只表示本实现检查了固定加载条件，报告始终保留 `target_player_run=not_executed` 和 `touch_playtest=not_executed`。下一步须在手机／平板真实导入与游玩，并记录Phira版本、设备、设备offset、同步、四键型与多指体验。
