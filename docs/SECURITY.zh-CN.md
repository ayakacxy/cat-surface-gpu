# 安全与漏洞报告

[English](SECURITY.md) · [简体中文](SECURITY.zh-CN.md)

## 支持的版本

安全修复面向最新稳定版。报告问题前，请先升级到最新版本，确认问题尚未修复。

## 报告漏洞

通过 [GitHub 私密漏洞报告](https://github.com/ayakacxy/cat-surface-gpu/security/advisories/new)提交。请勿在公开 issue 中披露尚未修复的漏洞。

报告应包含受影响版本、运行环境、复现步骤和预期影响。请使用合成输入，并移除日志中的凭据和个人信息。可同时提供最小复现示例或修复建议。

维护者会审阅报告，并与报告者协商修复和公开披露事宜。

## 数据处理

CAT-Surface GPU 在本地处理表面文件，不收集遥测信息，也不上传输入数据。GIFTI 文件由 nibabel 读取，计算使用 NumPy/Numba、CuPy 或 MLX，具体取决于所选后端。
