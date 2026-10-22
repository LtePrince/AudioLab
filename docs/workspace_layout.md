# 本地工作区与临时目录清理

2026-10-22 按用户要求移除 24 个已结束的 `AudioLab-*` Git worktree，保留一个主仓库
`Audio/AudioLab/`，当前检出 `develop`。六个长期 Git 分支继续保留，不为各分支长期
保留一套文件夹。`Audio/AudioLab_test_phira/` 用户提供的 Eltaw 谱包目录未移动。

## 当前产物位置

以下路径均相对主仓库 `Audio/AudioLab/`：

| 内容 | 当前路径 |
| --- | --- |
| 全 300 首训练的四组权重、缓存、曲线和预览 | `out/full_corpus_v1/` |
| 可转移的新权重 ZIP | `out/runtime_delivery_full300/audiolab-full300-history-20261022.zip` |
| 已安装的 H 组服务权重 | `checkpoints/full300_history/` |
| 其他历史实验、服务输出与两份未跟踪审查 JSON | `out/worktree_archive_20261022/<原工作目录名>/` |
| 逐文件来源、目标路径及 SHA 校验清单 | `out/worktree_archive_20261022/_migration/plan.json` |
| 迁移校验与最终清理记录 | `out/worktree_archive_20261022/_migration/result.json` |
| 原未提交文档的额外备份与补丁 | `out/worktree_archive_20261022/_migration/primary_uncommitted_files/`、`primary_uncommitted.patch` |

本次迁移保留 908 个文件（约 1.92 GiB），逐文件核对 SHA；92 个符号链接已核对并按
迁移目标修正必要的链接。原主仓库的三份未提交文档修改已重新应用，仍保持未提交状态。
模型、原始谱、缓存和结果报告内容未改；共享 `data/`、原有 `checkpoints/`、`.venv`
保留，临时目录里的 Python／pytest 缓存随 worktree 清理。

历史报告、缓存或脚本中的绝对路径保留生成时原文，不能再假设旧 worktree 目录存在。
可按迁移清单中的 `moves` 前缀映射找到现位置；重放旧命令时需要更新输入／输出路径。
保留的训练预览相对链接和音频链接在迁移后核对；旧聊天中指向已删除工作目录的链接
请改用上表位置。训练源代码可按记录的 Git commit 查阅，无需保留所有代码副本。

## 后续约定

日常修改在主仓库和对应模型分支进行，仅在确有运行隔离需要时创建临时 worktree。
完成后依次确认任务结束、保留未提交修改与有用产物、校验文件／链接、删除 worktree。
产物优先写入上述固定根目录，避免每次实验产生一个永久工作目录。清理不触发训练、
推理或模型优化，不改变 Eltaw 三项问题“仅记录、暂不修复”的状态。
