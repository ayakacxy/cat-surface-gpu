# 参与 CAT-Surface GPU 开发

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

修改应保持数值结果可靠，并让其他人能够复现修改带来的影响。

## 开发环境

先按照 [README 的安装步骤](../README.zh-CN.md#install)配置环境，将临时文件和缓存放在项目目录内。然后运行：

```bash
python -m pip install -e '.[cuda,test]'
python -m pytest -q --basetemp .work/setup/tmp/pytest
python scripts/verify_release.py
```

CPU 测试不需要 CuPy 或 GPU。CUDA 测试需要可用的 NVIDIA 设备；请单独说明跳过了哪些测试。

## 算法修改

说明修改了哪个上游计算步骤及其原因。除非修改本身明确采用不同算法，否则应保持网格编号、支持的参数和停止条件。执行失败时应明确报错，不要静默切换到另一种实现。

报告改进前，应使用相同输入和设置进行比较。报告三角形是否一致、坐标误差、求解器、精度、计时范围、重复次数和峰值内存。详细方法见[复现说明](REPRODUCIBILITY.zh-CN.md)。不要提交解剖数据或可识别个人身份的医学数据。

新增代码注释和 docstring 使用中文。第三方版权及许可声明保留原始语言。用户文档提供英文和简体中文版。

## 提交 Pull Request

每次修改保持明确范围，说明修改后的行为、相关数值差异以及已运行的测试。如果某项测试需要当前不可用的 GPU 或私有测试数据，请明确说明。注明改编代码的来源，并保留相关作者和许可声明。
