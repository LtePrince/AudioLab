# 完整起音分数作为软条件的训练实验

这条独立入口不修改产品默认或旧权重。A=`none`、B=`soft` 使用相同解码器架构、旧权重、样本顺序、优化步数和随机种子；额外无 bias 投影零初始化。音频特征始终保留。B 接收冻结 transcriber 的每帧概率与时间偏移，A 接收全零；不做阈值/NMS，不用目标谱生成该条件。两者均自由采样 DT/NOTE，不传强制 skeleton。

当前版本只接受与 memory 同率的旧 lane head：一帧 `1024/22050` 秒，偏移单位为该帧。音频理解沿用 1024 mel 帧核心＋128 上下文；各路线声学 encoder 保留 full attention。三前端均冻结，只计算一次缓存，训练和生成读取同一表示。高帧率 head 显式拒绝，避免静默丢失子帧。

```sh
python -m script.train_soft_onset prepare --manifest /path/manifest.json --model-root /path/timing_v1 --output out/soft-onset/cache --device cuda:0
python -m script.train_soft_onset train --cache out/soft-onset/cache --mode 4k --condition soft --steps 200 --seed 42 --lr .00005 --output out/soft-onset/4k-soft --device cuda:0
```

`mode` 分别用 `4k`、`freex`，`condition` 分别用 `none`、`soft`。短 pilot 可用 `--steps 32 --train-limit 2 --val-limit 1 --generate-limit 0`；正式实验从共同旧权重重新开始，不接着 pilot 权重训练。缓存清单每行须包含 `id/split/chart/audio/gameplay`，顶层 `source_root` 用于核对挖掘来源。

默认整曲不超过240秒、完整参考4096 token；超限拒绝，不截断音符/Hold。训练只保存一个 decoder final，无 optimizer/逐轮快照。默认RSS上限4GiB、空闲RAM/磁盘各8GiB、CUDA份额.65；缓存及运行目录另限700MiB。实验 checkpoint 用独立 schema，旧产品加载器明确拒绝；显式 loader 严格检查参数与条件。

每次训练结束评估全部指定开发留出，按分项 token bits 和当前NOTE的连续offset误差报告；默认前两首留出自由生成，保留 raw/final、自然EOS或预算失败、起点/曲长、四键型和现有pattern统计，并给出带音频的预览。参考起点F1只描述同曲编排相似度，不能当声学真值或可玩性认证。固定200步，不用留出挑训练时长。

第二阶段条件 `soft_history` 在B基础结构上加入[28维因果历史输入](causal_chart_history.md)，仍从共同旧权重重新训练200步，不在B结果上追加训练。新分支零初始化；构造模型之后重新设置训练RNG种子。Teacher forcing只读取更早token/连续offset，采样只读取更早生成值；当前NOTE的连续坐标在当前forward与offset预测完成后才进入状态，因此不能读自己的回归目标。该阶段不添加pattern辅助损失，不强制重复/交互/对称。
