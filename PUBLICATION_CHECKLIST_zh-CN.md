# GitHub 发布前检查表

本文件供作者正式创建公开仓库时逐项确认。

- [ ] 两位作者已同意以 MIT License 公开代码、合成数据、结果和图片。
- [ ] 仓库名称采用 `hierarchical-pomdp-ipso-multi-uav` 或在 README 中同步更新新名称。
- [ ] 没有加入 Word 稿件、审稿回复、真实坐标、现场日志、账号、令牌或私钥。
- [ ] 保留 README 和数据卡中“reconstructed synthetic reproducibility benchmark”的声明。
- [ ] 在本地运行 `python -m pytest`，全部测试通过。
- [ ] GitHub Actions 的 `repository-integrity` 工作流通过。
- [ ] 在仓库 About 中填写英文简介和 README 建议的 topics。
- [ ] 启用 Issues；如接受外部贡献，可启用 Discussions。
- [ ] 论文正式发表后，将 DOI、期刊、卷期页码和仓库 URL 补入 `CITATION.cff`。
- [ ] 创建版本标签和 Release（建议 `v1.0.0`），在 Release 中上传本成品 ZIP。
- [ ] 如需长期归档与可引用 DOI，可在作者确认后关联 Zenodo 等科研归档服务。

注意：创建 GitHub 仓库、推送代码、创建 Release 和关联外部归档都会改变外部状态，应由作者确认后执行。
