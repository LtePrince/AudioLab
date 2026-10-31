# AR 关键轮次与续训断点存档

保留 2026-10-26 运行包发布后的关键训练权重，供恢复与对照；不是新模型训练或质量认证。

- 四组：AL-AR4K v0.5.1 A/S、AL-FreeX v0.4.0 A/S，各 1100/1200/1300/1400 轮。
- free-x A 额外保留 1150 轮阶段最佳；新歌 E20 采用 A1300，其余三组采用 1400。
- 四个 ZIP 保存原始紧凑检查点，按仓库相对路径解压；不包含音频、谱包或冻结前端。
- 四个 resume_e1400.pt 是最新完整断点，包含 AdamW、采样器与 Torch/CUDA RNG，保留续训能力。
- manifest.json 记录每份原文件 SHA256、轮次和训练曲成绩；SHA256SUMS.txt 校验本 Release 附件。

这些是研究恢复文件，不能直接交给 script.model_bundle install。浏览器一键运行仍使用 anchored-ar-20261026 的原 B03/B04 安装包；没有替换已发布运行权重。

训练曲拟合不代表泛化通过：E20 十首新歌仍有时间跳转、段落缺音与停止问题。归因报告位于 docs/evaluations/phira_gap_diagnosis_20261030.md。

额外对照：AL-AR_additional_e1350.zip 保存 4K S1350 与 free-x A1350；两者在固定 15 首训练曲上的起点 F1 略高，但主联合 F1 低于各自首选。附加包的原文件 SHA 位于包内 supplementary-manifest.json，包校验见 SHA256SUMS-additional.txt。
