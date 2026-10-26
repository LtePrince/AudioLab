# 可迁移的推理模型包

当前长训实验权重及独立 A 对照包见 [2026-10-26 安装说明](anchored15_runtime.md)；本页下方保留旧 timing-v1 包记录。

2026-10-22 的 300 首训练实验权重另见 [新权重导入与浏览器测试](full300_runtime.md)。
以下仍是旧基线包的说明，两者通过不同 `--model-root` 切换。

另一台 Linux／NVIDIA PC 需要 Git 代码和单独转移的模型包。`checkpoints/` 不提交 Git，
没有已发布的公共权重下载地址。运行端不需要训练集、特征缓存或原开发机目录。
安装、驱动与服务命令见 [Linux 运行说明](linux_runtime.md)。

当前交付为 `audiolab-models-timing-v1-20261017.zip`，**133,452,353 字节（127.27 MiB）**。
SHA-256：

```text
64b7e43b82564c6a400cec38431918ea8f27c240ddf884471ae2d7d981a823f4
```

完整机器可读记录在 [runtime_model_manifest.json](runtime_model_manifest.json)。
本次开发机实际产物是原工作区 `out/runtime_delivery_v1/` 中的上述 ZIP；请将它单独复制到
运行 PC。这是现有三套基线权重的运输包，未重新训练，不表示声学质量或真实可玩性已验收。
当前音频权重的标签来源仍是 `chart_onset`。

## 运行 PC 导入

在仓库根目录创建 `out/runtime-models/`，将 ZIP 复制到该目录后执行下面命令。
该目录被 Git 忽略，避免将大包误提交或使生成报告误记代码有改动。这些工具只依赖
Python 标准库，可以先导入权重，再安装推理依赖。

```bash
python3 -m script.model_bundle verify \
  --bundle out/runtime-models/audiolab-models-timing-v1-20261017.zip \
  --sha256 64b7e43b82564c6a400cec38431918ea8f27c240ddf884471ae2d7d981a823f4
python3 -m script.model_bundle install \
  --bundle out/runtime-models/audiolab-models-timing-v1-20261017.zip \
  --model-root checkpoints/timing_v1 \
  --sha256 64b7e43b82564c6a400cec38431918ea8f27c240ddf884471ae2d7d981a823f4
```

成功后得到：

```text
checkpoints/timing_v1/
  model_bundle.json
  transcriber/transcriber_best.pt
  decoder/decoder_best.pt
  decoder_freex/decoder_best.pt
```

已有目标目录会被拒绝，包括空目录和软链接。升级时安装到新的目录，再使用启动入口的
`--model-root` 选择新版本；不要覆盖正在服务的权重。将 SHA-256 与代码中固定的交付记录
比较，确认转移的是同一份包。仅比较包内自带哈希不等于验证来源。

## 开发端导出

在需要更新实际权重时，从对应检查点根目录导出新文件，并记录新的版本和哈希：

```bash
python3 -m script.model_bundle export \
  --model-root checkpoints/timing_v1 \
  --output out/models-new-version.zip \
  --bundle-id new-version
```

工具只收录固定的三份推理检查点及清单，不收录数据集、训练缓存、优化器的独立文件或
绝对路径。检查点本身按原字节保留；因此导出前仍需确认所选检查点是要交付的版本。
同一 ID 和同一输入会产生相同 ZIP 字节。现有导出文件也不覆盖。

导出、导入按 1 MiB 块处理，并预留 **8 GiB** 磁盘余量；单权重上限 512 MiB、三权重
总上限 1 GiB。当前包加展开文件约占 255 MiB，另需留出依赖、上传和生成结果空间。
检查包括固定成员列表、普通文件类型、长度、ZIP CRC 和每文件 SHA-256。失败时删除本次
暂存文件，不发布半套权重。发布采用同一文件系统内的原子操作；不承诺断电恢复。

`model_bundle.json` 中的 `timing_schema` 是运行契约要求；运输工具不会反序列化 Torch，
`validation=file_integrity_only` 明确仅证明文件完整性。启动预检继续检查实际检查点和音频
读取路径；数学／音乐正确性和目标 Phira 手机／平板操作仍按各自指标验收。

传输专项回归覆盖原字节往返、无第三方包验证、目录／文件不覆盖、成员路径与类型、大小
上限、哈希错误、运行中余盘不足及失败暂存清理。资源失败使用模拟值，不耗尽真实磁盘。
