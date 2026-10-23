# 参与 CAT-Surface GPU 开发

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

欢迎报告问题、改进文档和提交代码。较大的算法或 API 调整，请先开 issue 讨论方案。

## 配置开发环境

在虚拟环境中使用 Python 3.11 或更新版本。从源码目录以可编辑模式安装项目和开发工具：

```bash
python -m pip install -e ".[test]"
```

开发 NVIDIA CUDA 后端使用 `.[cuda,test]`，开发 Apple Silicon Metal 后端使用 `.[metal,test]`。后端要求见[安装指南](../README.zh-CN.md#install)，环境和缓存设置见[构建指南](BUILDING.zh-CN.md)。

## 修改要求

每个 Pull Request 尽量只解决一个问题。用户文档的英文和简体中文版应同步更新。新增代码注释和 docstring 使用中文，第三方声明保留原文。

算法修改应说明对网格编号、求解器设置、精度和停止条件的影响，并在相同输入与参数下比较结果。报告拓扑检查结果和数值误差；如果涉及性能，还应说明硬件、计时范围、缓存状态和多次测量结果。具体方法见[复现指南](REPRODUCIBILITY.zh-CN.md)。

不要向仓库提交患者数据、访问凭据、本地缓存或构建产物。改编代码应注明来源并保留许可声明。

## 检查修改

按修改内容选择检查项目。仅修改文档时，检查链接和示例即可，无需运行 GPU 测试。

代码修改通常需要执行：

```bash
ruff check src tests tools scripts setup.py
ruff format --check src tests tools scripts setup.py
python -m pytest -q --basetemp .work/setup/tmp/pytest -o cache_dir=.work/setup/cache/pytest
```

涉及特定后端的修改，还应在对应硬件上测试。如果没有相关设备，在 Pull Request 中注明未运行的检查。打包相关修改请参阅[构建指南](BUILDING.zh-CN.md)。

## 提交 Pull Request

说明要解决的问题、修改后的行为以及检查方法，关联相关 issue，并附上必要的数值或性能结果。[英文 Pull Request 模板](../.github/pull_request_template.md)提供了基本格式。
