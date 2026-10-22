# O2：一首整曲的布局声学特征对照

这是自由采样实验入口，不改变产品默认，不重新选取音频候选、不做 teacher forcing 或训练。使用已完成的一个双模式产品目录，运行四次布局解码：4K/free-x × matched/zero-acoustic。首轮选 `product_smoke_full_v1/song-0`（102.770s、188 BPM、offset 0、seed42）。

```bash
python -m script.ablate_layout_features \
  --source-job ../AudioLab-product-development/out/product_smoke_full_v1/song-0 \
  --model-root ../AudioLab/checkpoints/timing_v1 \
  --output out/layout_feature_ablation_v1 --device cuda:0 \
  --threads 2 --max-rss-gib 4 --min-free-disk-gib 8 \
  --min-available-ram-gib 8 --gpu-memory-fraction .65
```

GPU选择由外部调度；若使用已有监督入口，`python -m script.train_models --gpu 1 --worker script.ablate_layout_features -- <上述参数>` 会在单worker子进程中映射选定GPU。开发验证不要求另装环境或复制权重。

输入音频、已有候选与权重均验证哈希；在同一音频mel上使用完整encoder。两条件分别重算encoder，要求源feature和valid mask指纹完全相同，seed重新置为同一值。零条件复用C2 `alter_features`，在 `decoder.prepare_memory` **之前**把encoder feature置零；decoder投影bias、时间嵌入、mask、固定候选时间与生成过程自己的历史保留。不是把memory或整个条件输入清空。

共同条件为 temperature1、top_p.95、flat采样、无押数上限、自然EOS、默认12000解码step预算。底层 `decode_notes(feature_condition=None)` 的原默认路径不执行干预，新增实验只由此脚本显式调用。两路线保留四键型与同一 `static-v1` 修复，不改Hold配额或未来起点规则。

报告保存每次源／干预feature、mask与token指纹，raw/final JSON、事件／修复日志、O1描述器、原冻结baseline的字节比较。两条件生成的音符数量可能不同，不能拿同序号当相同音符；对照只配对固定候选起点组。时间、位置、键型、持续的变化说明采样敏感性，不是音乐对应或可玩性改善。

输出最多5 MiB。两个条件目录各提供音频同步预览，音频为指向原产物的软链接，因此这些是本机评审产物，转移时需保留引用；本实验不生成新的Phira发布包。未成功的条件保留四次试验分母，整项不标成功；写后读回和最终资源检查失败会清理完成报告与预览。

先执行 `python -m pytest -q test/test_layout_feature_ablation.py test/test_product_pipeline.py`：小型真实decoder验证默认与matched一致、零特征前置而bias保留、奇数mel尾部、自然EOS预算、两模式完整产物与失败回收。小型随机权重只验证工程接口。原始曲目的真实实验结果另记，不用这些测试替代模型证据。
