# 全 300 首历史模型：个人 PC 测试

用户测试反馈：[Eltaw 的长条衔接、前半段 Tap 集中与节奏不稳定](evaluations/eltaw_user_feedback_20261022.md)
（2026-10-22，仅记录，按用户要求暂不修改）。

本包是 2026-10-22 完成的 H 组：完整软起音输入＋因果历史，4K 和静态无轨各一套。
300 首全部参与训练，各 10 遍；不含 D 组关系辅助损失。属于实验候选，保留已知提前
结束、起点匹配不稳定等问题，未替换默认 timing_v1 权重。训练结果见
[全量训练报告](evaluations/full_corpus_training_v1.md)。

## 导入与启动

先结束旧服务，在运行 PC 的仓库根目录、原 Python 环境中更新 develop：

```bash
git switch develop
git pull --ff-only origin develop
```

将单独转移的 `audiolab-full300-history-20261022.zip` 放进 `out/runtime-models/`。
包大小 **133,517,734 字节（127.33 MiB）**，包含两套新解码器、它们原有的冻结音频
编码器，以及训练时使用的起音模型。无需训练集或缓存，也不要直接覆盖旧 decoder 文件。

```bash
python -m script.model_bundle install \
  --bundle out/runtime-models/audiolab-full300-history-20261022.zip \
  --model-root checkpoints/full300_history \
  --sha256 be354e2acc358c09e6a790499e0e12639aa860f6f39f487f077e88aec1a596d8

python -m script.run_product \
  --model-root checkpoints/full300_history \
  --output out/local-full300-history \
  --min-available-ram-gib 4
```

打开 http://127.0.0.1:7860，照常上传 WAV／OGG，选择 4K 或静态无轨、预览或下载 Phira 包。
`--min-available-ram-gib 4` 针对运行 PC 先前因 8 GiB RAM 预留而启动失败的情况；
它改变的是系统内存预留，显存和进程内存保护保持原设置。服务每次启动都需要传入上述
`--model-root`；不传时仍加载旧的默认 timing_v1。启动报告应显示
`bundle_id: full300-history-20261022` 和两路线的 `quality_status: experimental`。

旧权重仍在原目录。要对照旧版，结束新服务后运行：

```bash
python -m script.run_product --model-root checkpoints/timing_v1 --min-available-ram-gib 4
```

## 加载与评价约定

产品入口仅对带有 `audiolab-soft-onset-runtime-v1` 合同的运输权重启用新路径。
研究训练文件仍不能直接用于旧加载器。新路径严格加载原参数，并验证冻结起音模型
指纹；每帧概率与偏移全部输入网络，无阈值筛选或强制 skeleton。局部音频前端采用
训练时的 1024／128 mel 分块，全曲布局编码器采用完整 attention，特征进行训练缓存
相同的 fp16→fp32 往返。历史与 free-x 连续残差反馈使用原采样实现。

生成报告保留 `runtime_contract`、模型哈希、原始谱面和后处理记录。候选覆盖字段为
null，因为自由生成没有强制候选集合；自然 EOS 及格式检查不证明整曲音乐覆盖或可玩性。
本包不改训练参数、不提高起音阈值、不增加多指上限。

开发端产物位于 `AudioLab-full-corpus-runtime/out/runtime_delivery_full300/`，大文件
不进入 Git。可通过 `python -m script.export_soft_onset_bundle --help` 查看再导出入口。

## 交付验证

92 项相关检查通过，覆盖旧产品路径、新权重严格加载、低概率起音保留、禁用强制骨架、
包传输、资源设置和失败处理。最终合同收紧后复跑 5 项新路径测试通过。
开发机启动预检在 RTX 3090 上返回 ready；使用安装后的本包，对 153.53 秒
`-SURREALISM-_IN`（BPM 180、offset 0、seed 42）运行完整产品管线，两路线及 Phira
包均完成，4K 992 音符、无轨 728 音符。两份原始谱面与原训练实验预览逐字段一致。
耗时 39.27 秒、进程峰值 RSS 1.71 GiB，仅是该开发机样例，不代表 4060 速度或质量验收。
未执行目标 PC 浏览器交互或手机／平板试玩。
