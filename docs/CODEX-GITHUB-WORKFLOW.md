# GitHub Codex 协作流程

本仓库采用 GitHub 官方 `@codex` PR 协作方式，不给自动化账号保留长期 `push` 权限。

## 使用方式

1. 将 ChatGPT 账号连接到 GitHub；
2. 从 `main` 创建功能分支并提交改动；
3. 打开 Pull Request；
4. 在 PR 描述或评论中提及 `@codex`，并明确请求 review、修改或问题分析。

例如：

```text
@codex review this PR for numerical-contract, licensing, and reproducibility issues.
```

Codex 的反馈和修改应通过 Pull Request 进行审查；合并仍由仓库维护者决定。真实数据、
本机路径、编译产物和未审查的外部依赖不得通过 Codex PR 带入仓库。

## 当前仓库边界

- `ayakacxy` 保留仓库所有权和合并权限；
- `OpenAI Codex` 作为 AI-assisted contributor 记录在 `AUTHORS.md`；
- `@codex` 的 GitHub 参与通过 PR 触发，不把个人访问 Token 或长期协作者权限交给自动化账号；
- 数值合同、许可证和最终发布仍由项目维护者负责确认。
