# 在另一台 Linux／NVIDIA PC 运行 AudioLab

当前电脑负责开发和验证，实际服务由另一台个人 PC 运行。运行端需要 Git 代码、独立 Python 环境和单独转移的模型包；不需要训练集、特征缓存、开发机路径或 `.venv` 软链接。首版提供定轨 4K 和静态 free-x，均保留 Tap、Drag、Hold、Flick 与多指；模型质量与 Phira 真机操作仍需分别验收。

## 准备运行环境

本轮基线为 Linux、Python **3.13**、PyTorch／TorchAudio **2.10.0**、TorchCodec **0.10.0**。其他关键数值依赖固定在 [requirements/runtime.txt](../requirements/runtime.txt)。它锁定直接／关键依赖，传递依赖仍由 pip 解析，**不是完整的跨平台锁文件**。不修改训练用 `pyproject.toml`，运行端不需要 `pip install -e .` 或 `uv sync`。

先安装 Git、Python 3.13 及其 venv 支持、可工作的 NVIDIA 驱动、FFmpeg 共享库。Python 3.13 是否由默认系统软件源提供取决于发行版，下面命令假定 `python3.13` 已安装。Debian／Ubuntu 的媒体与 Git 依赖可用：

```bash
sudo apt install git ffmpeg libsndfile1
```

PyTorch 的 [Linux 安装说明](https://pytorch.org/get-started/locally/)要求受支持的 Linux／glibc，并提供 CUDA 选择方式；当前页面说明 glibc 至少 2.28。[2.10 官方安装矩阵](https://pytorch.org/get-started/previous-versions/#v2100)提供 cu126、cu128、cu130 构建。必须结合该 PC 的驱动和 GPU 选择，不能把开发机的 CUDA 构建当作所有 NVIDIA 显卡的通用选项。`nvidia-smi` 可用于检查驱动与显卡；后面的实际 CUDA 小运算继续检查运行可用性。

```bash
git clone --branch develop https://github.com/LtePrince/AudioLab.git
cd AudioLab
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

下例选择 CUDA 12.8 构建，与当前开发基线相符；只有目标显卡／驱动支持时才使用。需要其他构建时，将官方矩阵中的索引用于同版本 `torch` 和 `torchaudio`，不要混装两个版本：

```bash
python -m pip install --no-cache-dir --only-binary=:all: \
  torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir --only-binary=:all: -r requirements/runtime.txt
```

第二条命令保留满足 `2.10.0` 的已安装 CUDA 构建，包括 `+cu128` 等本地版本后缀。`--only-binary` 使没有合适 wheel 的平台明确失败，不自动进入大型源码编译。若 pip 找不到匹配 wheel，应核对 Python／系统架构和官方构建支持，再调整环境。

[TorchCodec 0.10 发布说明](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.10.0)确认与 Torch 2.10 配套。[其安装说明](https://github.com/meta-pytorch/torchcodec/tree/v0.10.0#installing-torchcodec)要求 FFmpeg 共享库，支持主要版本 4–8；仅有一个独立的 `ffmpeg` 可执行文件不保证能被 Python 加载。[TorchAudio 2.10 的音频读取接口](https://docs.pytorch.org/audio/2.10.0/generated/torchaudio.load.html)通过 TorchCodec 解码，因此不能省略这项运行依赖。绘图只在开发调试分支按需导入；运行清单无需 OpenCV、TorchVision、Transformers、Matplotlib 或 pytest。

## 导入模型并启动

`git clone` 不含权重，也没有自动下载模型的步骤。将独立交付的 `audiolab-models-timing-v1-20261017.zip` 放到被 Git 忽略的 `out/runtime-models/`，按[模型包导入说明](model_bundles.md)验证固定 SHA-256 并安装到 `checkpoints/timing_v1`。该路径包含三份权重和 `model_bundle.json`；安装时不会覆盖已有目标目录。导入工具只用标准库，可在安装推理依赖之前运行。

在仓库根目录、已激活的运行环境内执行：

```bash
python -m script.run_product --check
python -m script.run_product
```

第一条成功时输出 JSON，`status` 为 `ready`。第二条先做相同预检，通过后在前台启动服务。运行 PC 的浏览器打开 `http://127.0.0.1:7860`，选择 WAV／OGG 与玩法，生成后试听预览并下载完整 Phira ZIP，再转移到手机／平板本地导入。按 `Ctrl+C` 停止服务；已有任务记录和结果保留。服务只绑定 loopback，本说明不包含公网或局域网部署。

默认模型目录是 `checkpoints/timing_v1`，输出是 `out/local-service`。相对路径按执行命令时的当前目录解析；启动前请 `cd` 到仓库根目录。显式路径支持空格：

```bash
python -m script.run_product \
  --model-root "/path/with spaces/models" \
  --output "/path/with spaces/results" --port 7863 --device cuda:0
```

默认 `--device cuda` 使用当前可见的 CUDA 设备，预检解析到明确编号后把同一个编号传给服务／worker。入口保留用户已有的 `CUDA_VISIBLE_DEVICES` 映射，不强制选择物理 GPU 1，也不经过训练启动器。显式 `--device cpu` 可用于 CPU 验证；CUDA 不可用时不会自动退回 CPU。CPU 的整曲速度和容量未承诺。

## 预检做什么

预检先用标准库检查 Linux／Python、依赖元数据、三个模型文件、磁盘／RAM、可写输出目录、目录占用与本地端口。发现问题会返回非零退出码，JSON 的 `error` 包含阶段、原因和处理建议，不会启动网页服务。

模型包 manifest 存在时，复用运输工具验证有界清单和每份文件的大小／SHA；然后顺序在 CPU 用 `weights_only=True` 实际加载三份权重，拒绝旧时间约定、损坏状态和不符的 4／64 位置布局。兼容显式指定的旧开发权重目录，但没有 manifest 时报告 `unbundled`，不伪称已验证运输来源；权重须来自可信交付。

随后实际走短 WAV／OGG 的解码、48 kHz→22050 Hz 重采样与 mel 路径，并运行小型节奏估计夹具和选定设备上的矩阵运算。报告记录 Python／关键依赖、文件指纹、媒体时长差、设备编号／名称／CUDA 构建及资源。探针不加载训练数据或写特征缓存。首次调用依赖自身可能生成小型 Python／Numba 缓存，预检临时媒体会清理。正式启动用 `exec` 替换预检进程，释放其模型与 CUDA 状态，再由轻量服务串行启动生成 worker。

预检只证明这些运行路径可执行。当前开发环境对自产 0.25 秒 Vorbis 探针解码出了约 0.268027 秒，报告保留约 18.03 ms 的差值；WAV 差值为一个分析样本内的取整。该 OGG 结果证明媒体库可工作，**不表示每种 OGG 编码都能通过现有严格的谱包时长检查**。本轮没有改音频特征、放宽打包容差或隐藏该现象；遇到此类文件可先尝试相同音频的 WAV，并保留失败记录供后续修复。

## 资源与验证边界

运行与模型安装保留至少 **8 GiB 磁盘余量**；启动另检查少量服务状态空间。上传阶段还按音频、快照和两种谱包预留空间。RAM 检查使用当前可用值，要求至少 **8 GiB 可用**；生成进程 RSS 上限 **4 GiB**，CUDA allocator 预算比例 **0.65**。这是运行预算，不是安装环境的总容量，也不是整卡显存的硬限制。安装前需另为 CUDA wheels、环境和下载留足空间。

短 CUDA 探针不能证明所有歌曲都装得下。全曲布局仍按输入长度预估注意力／解码器工作集，资源不够会明确失败，不静默换成分块布局。600 秒是输入上限而非该长度已通过实测的承诺。单服务一次只执行一个生成任务；并发请求、取消和输出目录锁沿用现有实现。

实现提交 `c948ca6`：20 项专项测试通过，设计 agent 在固定代码独立复验 **20 passed／9.07 秒**。测试覆盖无第三方包的帮助／缺依赖信息、坏模型／旧时钟、依赖版本混用、资源拒绝、媒体真实解码与可选包隔离、CUDA 选择／无自动 CPU 回退、带空格路径和目录／端口占用。GPU 调用与资源不足边界使用模拟，不制造真实 OOM。

开发机因磁盘预算复用既有 Python 环境，没有另装一套大型 CUDA 环境；目标 PC 的全新 pip 安装、驱动和实际性能尚未执行。主线程的隔离代码副本／模型导入／实际网页验证记录在[产品集成记录](product_integration.md)，应与真正另一台 PC 的安装结果区分。Phira 手机／平板导入和触屏手感也继续单独验证。
