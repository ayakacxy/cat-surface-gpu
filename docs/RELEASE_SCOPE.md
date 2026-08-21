# Release scope

这是从 Fast CHARM 工作树整理出的独立 CAT-Surface GPU 发布候选目录。
发布目标是让外部用户可以审查加速实现、运行小型测试并按自己的 CAT-Surface 构建复现实验。

## 纳入内容

- `src/fast_charm/cat_surface/`：GPU 曲面、旋转、DARTEL、warp 和 `CAT_Surf2Sphere` 实现；
- `tools/run_cat_*.py`、`tools/benchmark_cat_*.py`、`tools/audit_cat_warpsurf.py`：运行和 A/B 工具；
- `tools/cat_surface_c/`：需要外部 CAT-Surface headers/library 的 C helper 源码；
- `tests/`：不依赖真实被试数据的单元和拓扑合同测试；
- `upstream/`：上游 CAT-Surface 版本和许可证说明；
- `README.md`、本文件和后续补充的复现文档。

## 明确排除

- 当前 Fast CHARM 的 GEMS、SAMSEG、ITK 和完整 SimNIBS 导入树；
- `third_party/CAT-Surface/` 的完整上游源码快照；公开仓库只记录版本并要求用户自行准备依赖；
- 任何 `*.o`、静态/动态库、CAT/SimNIBS 可执行文件、CUDA 缓存和 profiling 输出；
- 真实被试数据、GIFTI/NIfTI/MGH/MESH 输入输出、`/tmp` 结果和本机 benchmark JSON；
- 任何本机绝对路径或临时目录路径；
- 未经过发布审查的 SimNIBS `brain_surface.py` 整文件。官方集成如需公开，应另做最小补丁并单独核对许可证。

## 当前上游基线

- CAT-Surface 算法参考 commit：`628b6851d8638f3ab773cd25c0ec406d0ec61ede`；
- 当前实测环境：RTX 2080 Ti、Torch `2.6.0+cu126`、Triton `3.2.0`；
- 当前默认严格球面路径：`areal_schedule=ordered`、`areal_arithmetic=cat`；
- reference 后端长期保留，GPU 后端只通过显式 runner/参数启用。

## 许可证决定

- 本仓库新增代码：`GPL-3.0-or-later`；
- CAT-Surface 及其第三方组件：继续遵守 `upstream/` 和外部依赖自身许可证；
- 不把 `genus0` 或其他未明确授权的上游组件复制进本仓库；
- 如果未来加入 SimNIBS 集成补丁，仍需保留 SimNIBS 的 GPLv3 声明并重新检查组合分发边界。

## 发布前门槛

1. 确认新增代码和外部 CAT-Surface/SimNIBS 依赖的许可证兼容性；
2. 补齐正式作者、维护者、版权和引用信息；
3. 用不包含真实数据的 fixture 做干净环境测试，并记录 GPU 型号、Torch/CUDA/Triton、输入形状、
   计时边界、同步方式、峰值显存和误差；
4. 对公开文件执行绝对路径、敏感数据名、二进制和生成物扫描；
5. 由维护者确认 GitHub 仓库地址、可见性、默认分支和首个 tag 后再初始化提交并推送。
