# GitHub 公开发布说明（中文）

本目录是论文 **“A Hierarchical Dual-Stream POMDP–IPSO Framework for Multi-UAV Cooperative Inspection Scheduling and Control”** 的 GitHub 公开仓库成品包。

## 发布前必须理解的数据边界

本仓库公开的是为了复现而构建的 **19 个场景、574 个任务的合成基准**，不是现场实测数据。现有材料中没有可公开的原始运行坐标、原始仿真器、训练检查点或现场日志，因此这些内容没有被虚构或替代。公开时请保留英文 README 和数据卡中的醒目声明，不能把合成坐标描述成真实输电线路坐标。

## 已整理内容

- `scripts/`：绘图、调度基准和控制器基准程序；
- `data/`：合成任务数据及随机种子、预算、环境元数据；
- `results/`：逐次运行结果、汇总统计和检验结果；
- `figures/`：600 dpi PNG 和矢量 PDF；
- `docs/`：复现步骤和结果解读；
- GitHub 标准文件：许可证、引用信息、贡献指南、行为准则、安全政策、Issue/PR 模板和 CI 检查。

## 建议的 GitHub 仓库信息

- Repository name: `hierarchical-pomdp-ipso-multi-uav`
- Description: `Reproducible synthetic benchmarks and code for hierarchical POMDP-IPSO multi-UAV inspection scheduling and control.`
- Suggested topics: `multi-uav`, `pomdp`, `mappo`, `particle-swarm-optimization`, `task-scheduling`, `path-planning`, `reproducible-research`
- Visibility: `Public`

上传前请先确定作者对代码、图和合成数据拥有公开授权。论文正式发表后，应在 `CITATION.cff` 和英文 README 中补充论文 DOI、期刊信息、最终年份及真实 GitHub 仓库地址。

## 最简复现命令

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/generate_revision_figures.py
python scripts/run_scheduling_benchmark.py
python scripts/run_controller_benchmark.py
```

控制器训练建议使用 CUDA GPU；仓库已经包含全部结果，读者无需重新训练即可核查表格。

## 不应上传的内容

- 原始 Word 稿件和审稿回复信；
- 未获授权的现场坐标、线路名称或内部资料；
- 个人身份信息、账号、令牌、密码和私钥；
- 无法说明来源与授权的第三方图片、模型或代码。

本文件仅用于作者发布时的中文操作提示；英文 `README.md` 是公开仓库主页。
