# E27：Event4K取消分窗，整曲仅一个EOS

2026-11-05，按用户要求取消Event4K输出分窗。实现与CPU验证完成，**训练保持暂停，新协议没有自然歌曲训练权重**。原E26-BPM进程2391455复核为`Tsl`；已保存8轮，暂停前最后日志为2440步，最新保存身份仍为AL-Event4K v0.2.7。没有恢复或替换该进程中的旧模型。

## 修改及理由

原实现已经跨窗保留全部历史，但每10秒分别限制时间候选并监督END_WINDOW。因此“历史被截掉”和“学到周期性结束”是两件事；前者未在原实现中发生，后者是本次待验证的假设，不能在重训前宣称已归因。

新协议把一段完整音频及其N个事件编码为`BOS, E1, …, EN, EOS`，只有N+1次决策，前N次预测事件，最后一次预测整曲EOS。静音只有BOS→EOS。没有10秒调度、窗口时间掩码、中间结束标签或跨窗重启。

每次时间预测在全部剩余有效音频帧与整曲EOS之间作联合softmax选择。时间必须递增且在实际音频范围内；这不限制cross-attention，生成任何位置时仍能看完整音频。谱面因果self-attention保留所有已生成事件，整曲只初始化一次KV缓存。时间嵌入、可选BPM、4×5整组事件与Hold首尾表示均保留，没有新增手工历史摘要。

时间头的`end_head`改为`eos_head`，训练和评测使用`eos_count/eos_ce/false_eos_rate/eos_recall`，不再汇报窗口结束率。音频范围耗尽与事件预算耗尽是明确的强制停止，不能计为自主EOS；EOS时仍有未闭合Hold也报失败。提前EOS仍可能缺音，取消分窗不能保证该问题已经解决。

## 验证结果

34项单元与集成测试通过：包含跨10/20秒的全曲候选支持、历史缓存连续增长、EOS后不再开窗、预算耗尽不冒充EOS、静音、混合多押、Hold结束与重开、因果性、增量推理一致、全参数反向传播、旧标签拒绝和完整断点恢复。

受控拟合测试使用一次性小模型和1秒随机Mel，250次更新后损失小于0.01；自由生成恢复Tap＋Hold双押和后续Flick，三个音符起止误差小于2ms，最后自主EOS。该小例权重丢弃，不是自然歌曲训练或成谱质量证据。

全300首、1200份既有视图重新编码通过，保留326,277音符、287,291事件时刻、33,670个HoldEnd、55,562个多音符时刻。**去掉5,512个中间窗口结束标签，仅保留1,200个序列末尾EOS。** 最大往返误差`5.000089e-10秒`，零丢音、零不支持样本。原有两份24秒物理裁剪和静音仍是独立音频样本，不把完整歌曲拆窗生成；整曲训练视图保留全部历史。

复用原Mel缓存，只新增目标缓存52,581,348字节（约50.1MiB），审计后磁盘约剩25.76GiB。没有本次GPU训练或大模型资源试跑。数据、代码哈希及验证命令见[审计记录](assets/event4k_fullsong_20261105/codec_audit.json)。

## 版本和续跑边界

基础输入仍是4×5同步事件，保留模型名AL-Event4K。为不同输出协议预留**v0.3.0（整曲有BPM）**与**v0.4.0（整曲无BPM）**；两者当前`weights=null`。未来各自不同权重递增末位编号。参数量未变：有BPM为13,439,894，无BPM为13,439,638。

新schema为`audiolab-synchronous-event4k-full-song-v1`，默认输出目录`out/event4k_fullsong_bpm_20261105`。旧分窗目标、probe、checkpoint不能直接恢复为整曲协议；E26原代码由提交`28f2a2b`保留，旧权重、源代码哈希和暂停运行记录均不改写。暂停进程已经加载旧Python代码，恢复它不会自动使用新协议，因此当前不发送SIGCONT。

入口仍为`script.train_event4k`。本次只执行`prepare --device cpu`和测试。后续获准启动前，需对新协议单独做GPU梯度／资源预检，再用新目录从零训练；不能把旧窗口实验的预检当成新协议训练已验证。

```bash
.venv/bin/python -m script.train_event4k prepare --device cpu
.venv/bin/python -m pytest -q test/test_event4k.py test/test_event4k_training.py \
  test/test_joint_audio_ar.py test/test_joint_audio_scratch.py \
  test/test_windowed_ar.py test/test_windowed_freex.py
```
