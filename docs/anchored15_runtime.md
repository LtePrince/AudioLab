# 2026-10-26：4K / free-x 长训权重交付

[GitHub Release](https://github.com/LtePrince/AudioLab/releases/tag/anchored-ar-20261026)。
沿用此前的 ZIP → `script.model_bundle install` → `script.run_product` 工作流。
**只发布权重和校验文件；15 首评测歌曲及谱包不上传 Release。**

**运行端已支持截断预览。** 已安装本页权重的用户不用重新下载：停止服务，更新
`develop` 后按原命令重启，并重新生成。模型到音频末尾仍未结束时，保留已有音符，
结果页、预览页及任务历史显示警告，仍可下载 Phira 包。之前失败的任务需重新生成。

| 包 | 4K | 静态 free-x | 下载 |
| --- | --- | --- | --- |
| B03，平均表现优选 | AL-AR4K v0.5.1 / S / E14-S / e1000 | AL-FreeX v0.4.0 / A / E18-A / e1000 | [audiolab-anchored15-20261026.zip](https://github.com/LtePrince/AudioLab/releases/download/anchored-ar-20261026/audiolab-anchored15-20261026.zip) |
| B04，纯音频 A 对照 | AL-AR4K v0.5.1 / A / E14-A / e0920 | 同 B03 | [audiolab-anchored15-a-20261026.zip](https://github.com/LtePrince/AudioLab/releases/download/anchored-ar-20261026/audiolab-anchored15-a-20261026.zip) |

每包约 **128.03 MiB**，包含两条路线的独立解码器、各自冻结音频编码器和起音模型，
无需训练数据或缓存。A 的额外起音通道按训练设置置零；S 输入全部连续起音概率及偏移。
没有将 A/S 混合训练，也没有新训练或更改这些检查点的参数。

## 在另一台 PC 上运行

先停止旧服务，在仓库根目录、原 **Python 3.13** 环境中操作：

```bash
git switch develop
git pull --ff-only origin develop
mkdir -p out/runtime-models
curl -fL --retry 3 -o out/runtime-models/audiolab-anchored15-20261026.zip \
  https://github.com/LtePrince/AudioLab/releases/download/anchored-ar-20261026/audiolab-anchored15-20261026.zip
python -m script.model_bundle install \
  --bundle out/runtime-models/audiolab-anchored15-20261026.zip \
  --model-root checkpoints/anchored15_20261026 \
  --sha256 813c3722e5e463e215ed2cdfdfdc73d184de4b4819843761d8c2afb8d48ac944
python -m script.run_product \
  --model-root checkpoints/anchored15_20261026 \
  --output out/local-anchored15 \
  --min-available-ram-gib 4
```

打开 **http://127.0.0.1:7860**，上传 WAV/OGG，照常预览和下载 Phira 谱包。
预览新增模型名称、条件和轮次，启动报告包含 bundle ID。
新任务的下载文件名使用「音频文件名＋玩法＋生成日期」，例如
`Eltaw_4k_20261026.phira.zip`、`Eltaw_freex_20261026.json`。
日期取运行服务的电脑在任务开始时的本地日期；中文文件名也可保留。
对照本次 Eltaw 结果时，在「可选设置」填 **193 BPM**；自动估速尚未完成质量验收。

测试 A 对照包时先停止服务，使用独立目录：

```bash
curl -fL --retry 3 -o out/runtime-models/audiolab-anchored15-a-20261026.zip \
  https://github.com/LtePrince/AudioLab/releases/download/anchored-ar-20261026/audiolab-anchored15-a-20261026.zip
python -m script.model_bundle install \
  --bundle out/runtime-models/audiolab-anchored15-a-20261026.zip \
  --model-root checkpoints/anchored15_a_20261026 \
  --sha256 efc5531742fc45ca1b5770c64bf3850550692dce9e629440637de50dd8948027
python -m script.run_product \
  --model-root checkpoints/anchored15_a_20261026 \
  --output out/local-anchored15-a \
  --min-available-ram-gib 4
```

安装器不覆盖已有模型目录；每次启动须指定对应 `--model-root`。
两个 ZIP 均低于既有 1 GiB 包限制；系统 RAM 预留显式设为 4 GiB，其他资源保护保持原设置。

## 选型与测试范围

先按既有三首评测，从仍保留的检查点中筛出每种玩法最佳 A、最佳 S 和一个额外候选；
再固定 Eltaw + 随机 14 首（seed **20261026**），比较六个候选的 15 首均值。
因此「优选」指本次候选比较结果，并非穷举每个权重在全部歌曲上的最高分。
这 15 首均参加过训练，不作为泛用性评估。

| 交付模型 | 平均联合 F1，未完成计 0 | 正常结束 |
| --- | ---: | ---: |
| 4K S / e1000 | 0.4984 | 10/15 |
| 4K A / e0920 | 0.3562 | 9/15 |
| free-x A / e1000 | 0.2749 | 10/15 |

联合 F1 要求起点在 50 ms 内且位置 bin、键型一致，分母包含全部音符。
网格仍只作匹配加分，不吸附时间、不要求固定排布；原始起点 F1、网格分与失败原因见
[六候选完整结果](evaluations/assets/runtime_selection15_20261026/summary.json)。
仍有无法正常结束的曲目，**本次为实验权重，未通过可玩性或泛用性验收**。

曲目：Eltaw、DESTRUCTION 321、Incyde、Diamond Dust、インフェルノシティ、The Whole Rest、
混乱-Confusion、Cipher 20、A journey to the moonlight、The Chariot REVIIVAL、风屿、Rrharil、
Rainy Heart、Re_Nascence Psystyle Ver.、GOODBOUNCE。训练目录 ID、音频指纹与固定抽样记录见
[selection.json](evaluations/assets/runtime_selection15_20261026/selection.json)。

## 运行兼容与验证

E14/E18 使用绝对时间 anchor/group 协议，不能直接替换旧 DT 解码器文件。
新运输合同严格加载原参数及冻结前端，复现全局音频、fp16 缓存往返、因果历史和连续坐标。
交付默认沿用本次贪心解码；音符起止时间不取整，且不启用旧产品的 Hold 裁剪或和弦位置修补。
当模型到音频末尾、解码步数上限或无法继续生成键型时，运行端保留可用前缀并提示警告。
截断时丢弃音频之外及尚未完成的音符，越过音频末尾的 Hold 尾裁到末尾；其余音符保留。
生成报告分别记录任务产物完成与模型未正常停止，保留原始错误、截断原因及修改数量。
原研究生成器和上述 benchmark 的自然停止指标保持不变；未产生有效音符仍不发布空谱。

100 项相关自动检查通过。两包均经过真实安装、文件校验和原检查点参数逐张量比对。
开发机 RTX 3090 上，B03 在 Firefox 实际上传 Eltaw 后完成两路线、播放、拖动、玩法切换、
下载哈希和窄屏检查；847 个 4K 音符、861 个 free-x 音符与评测逐项一致，
时间及坐标仅有不超过 1.5e-14 的单位换算误差。整条管线约 61 秒、进程峰值 RSS 1.71 GiB，
这些是开发机样例，不是目标 4060 的性能实测。验证记录见
[runtime_verification.json](evaluations/assets/runtime_selection15_20261026/runtime_verification.json)。
B04 的 4K A/e0920 另经实际安装后的整曲推理校验，847 个音符也与评测逐项一致；
B04 的 free-x 权重与 B03 文件哈希完全相同。

截断修复另经98项相关检查及Firefox实测：Eltaw自动估速129.19921875 BPM时，
4K未自然结束的952个音符可预览下载并显示警告，free-x正常结束的872个音符保留。
刷新后历史警告仍可见，切换玩法时显示对应警告；见
[cutoff_preview.json](evaluations/assets/runtime_selection15_20261026/cutoff_preview.json)。
