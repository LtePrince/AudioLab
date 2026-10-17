# Phira 完整谱包工程交付 v1

2026-10-17，`chore/phira-delivery` 的实现为 `f2e362f`，真实网页生成使用含共享 CUDA 预算修复的干净提交 `c3ef29a`。目标固定为 Phira v0.8.2；源码、格式、时间约定和资源上限见[打包说明](../phira_packages.md)与[引用清单](../phira_package_reference.json)。首版为四固定触屏位置的 4K 与静态 free-x，保留四键型与多指；本批没有改变权重、采样或 Hold 修复默认。

已有产物的 CPU 补包使用 `AudioLab-product-development/out/product_smoke_full_v1/song-0`，不重新推理。输出在本工作区 `out/phira_examples_v1`，完成报告 SHA-256 为 `85d9fefdbe89ae39120f283c870f4288443c5fbfdaeb2a9a7c16a6367bdeab79`。

| 模式 | 音符数 | ZIP 字节数 | ZIP SHA-256 |
| --- | ---: | ---: | --- |
| 4K | 735 | 1,604,011 | `712c62939da87c91ef1499012cac14d761716671cd7d42a086282f159776d072` |
| free-x | 783 | 1,612,926 | `4a69f0d42bec1dca236f4fe039fdc049e31575ab5db5fd158659b4f763a79f26` |

两包均包含原谱、原音频、封面和元数据，离开工作区后不依赖本地引用。设计复核了所有成员 CRC／哈希、原字节、PNG，以及 1,518 个音符的头尾、类型和位置；102.770068 秒音频已完整流式解码。补包后读回校验耗时 0.500 秒，校验进程峰值 RSS 为 0.0414 GiB，校验前输出为 3,223,140 字节；这些数值不是完整生成或打包阶段的峰值。独立证据为主工作区 `out/product_coordination/phira_package_artifact_review.json`。

独立包与管线测试 **41 passed**。主线程在 `develop` 集成点 `3f973de` 完整回归 **359 passed，9 warnings，42.18 秒**。首次网页任务因裸 `cuda` 设备参数在预算接口失败；`c3ef29a` 修复后用同一参数重新通过，旧失败记录保留且没有发布 ZIP。未以更换设备参数规避该问题。

真实 Firefox 从网页上传同一首音频，以自动时基生成双模式并实际下载：任务 `6afaf9a4ddb34a88bd200b6b962e2dfa`，生成及打包 31.8806 秒，输出 566／839 个音符。两份下载的大小和哈希与完成记录一致；Range 206、预览模式对应下载、刷新后的历史入口、390 px 页面均通过。独立解读下载文件再次核对了原谱／音频字节与全部 1,405 个音符。峰值 RSS 约 1.93 GiB，PyTorch 峰值 allocated／reserved 为 0.100／0.111 GiB，结束余盘约 11.67 GiB；这些显存统计不代表整卡占用，单曲耗时不构成速度承诺。

上述真实下载报告、具体 ZIP 指纹、服务恢复命令与合成校准材料统一记录在[产品集成记录](../product_integration.md)的“Phira 完整谱包与触屏试玩材料”一节（记录提交 `7d6d425`）。浏览器证据为主工作区 `out/browser_review/phira-workflow-v2.json`，独立下载证据为 `out/product_coordination/phira_browser_download_review.json`。24 秒／25 音符／非零 offset 的校准包明确标为合成，不能计入模型质量。

PH1–PH3 的完整包与工程下载交付通过；PH4 的 Phira 真机导入、触屏操作和整谱音乐体验尚未执行。当前入口为 `http://127.0.0.1:7862`，旧 7860 服务保留。本次只完成工程交付，保持 `target_player_run=not_executed`、`touch_playtest=not_executed`，下一步将校准包及同曲两种模型谱包交给用户在手机／平板实测。
