# CAT-Surface GPU acceleration

这是从 Fast CHARM 工作树中单独整理出的 CAT-Surface GPU 加速候选仓库。
它保留 SimNIBS 4.6/CAT-Surface 的 reference 路径，并通过显式参数启用优化后端；
GPU 不可用或 GPU kernel 失败时会直接报错，不会静默回退到 CPU。

当前仓库是公开发布前的 `0.1.0.dev0` 整理版，不应被描述为官方 SimNIBS 替代品，
也不包含完整 SimNIBS、SAMSEG、ITK、真实被试数据或任何预编译二进制。

## 包含的加速部分

- `CAT_Surf2Sphere`：保留官方顶点依赖顺序，在依赖层内向量化入射三角形面积、中心和
  面积加权更新；默认使用 `ordered + cat` 算术边界（float 存储、官方 double 累加边界）。
- 曲面 stencil：邻接表、稳定图着色、官方依赖层调度、拓扑 workspace 缓存和设备常驻数据。
- 初始旋转：规则球面候选定位、候选批处理、压缩候选索引和官方顺序 Nelder--Mead。
- DARTEL/warp：混合边界重采样、膜系统、松弛、平方更新、DARTEL 迭代、`expdef`、
  CUDA Graph 和 source/target CUDA stream 编排。
- 验证工具：真实 GIFTI 的 CPU/GPU A/B、阶段计时、输出拓扑检查和绝对误差合同检查。

核心 Python 实现位于 `src/fast_charm/cat_surface/`；必要的 CAT-Surface C helper 源码位于
`tools/cat_surface_c/`。仓库不携带 CAT-Surface 的完整上游源代码，使用者需要自行准备
匹配版本的 CAT-Surface reference/build，并遵守其许可条款。

## 已验证结果

以下是 RTX 2080 Ti、Torch `2.6.0+cu126`、Triton `3.2.0` 上同一真实粗球面输入的当前记录，
仅用于说明已测口径，不代表所有 GPU、输入和完整 CHARM 流程的固定收益。

| 阶段 | LH | RH |
| --- | ---: | ---: |
| 独立 `CAT_Surf2Sphere` GPU 墙钟 | 10.8119 s | 10.6667 s |
| 其中面积平滑 | 3.7084 s | 3.5142 s |
| 相对最新版 CPU reference 的顶点 max/mean/p99 | 1.14e-5 / 1.77e-6 / 7.63e-6 | 1.53e-5 / 1.68e-6 / 7.63e-6 |

SimNIBS 默认粗网格的双侧 GPU 阶段一次实测为 `55.2573 s`，其中球面阶段 `15.7609 s`、
官方 topology 上采样 `4.5208 s`、双侧 `CAT_WarpSurf` `34.9756 s`。最终 `sphere.reg`
面片拓扑 exact；LH/RH 顶点 max/mean/p99 分别为
`1.38e-6/7.09e-8/5.55e-7` 和 `1.81e-4/6.74e-6/5.70e-5`，均低于当前 `1e-3` 合同。

## 安装和测试

先安装与你的 CUDA 驱动匹配的 PyTorch，再安装本仓库：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6' 'numpy>=1.24' 'nibabel>=5.0'
python -m pip install -e '.[test,cuda]'
python -m pytest -q
```

上面的 CUDA/Triton 依赖是按当前验证环境写的最低候选版本；不同 CUDA、Torch 和 Triton
组合必须重新做数值 A/B，不应仅凭版本号宣称兼容。

## `CAT_Surf2Sphere` GPU runner

需要一个由最新版 CAT-Surface 源码构建的 CPU reference CLI。示例中的路径均为占位符：

```bash
export REPO_ROOT=/path/to/cat_surface_gpu
export CAT_SURF2SPHERE_REF=/path/to/CAT_Surf2Sphere

PYTHONPATH="$REPO_ROOT/src" python "$REPO_ROOT/tools/run_cat_surf2sphere_gpu.py" \
  --reference-cli "$CAT_SURF2SPHERE_REF" \
  --input-surface /path/to/input.gii \
  --output-surface /path/to/output.gii \
  --stop-at 10 \
  --device cuda \
  --kernel triton \
  --areal-schedule ordered \
  --areal-arithmetic cat
```

严格 A/B 使用：

```bash
PYTHONPATH="$REPO_ROOT/src" python "$REPO_ROOT/tools/benchmark_cat_surf2sphere_gpu.py" \
  --reference-cli "$CAT_SURF2SPHERE_REF" \
  --input-surface /path/to/input.gii \
  --gpu-output /path/to/gpu.gii \
  --reference-output /path/to/cpu.gii \
  --device cuda \
  --kernel triton \
  --areal-schedule ordered \
  --areal-arithmetic cat \
  --max-vertex-error 1e-3 \
  --max-mean-error 1e-3 \
  --max-p99-error 1e-3
```

## 完整曲面 pipeline 的边界

`run_cat_surface_gpu_pipeline.py` 还需要由 CAT-Surface headers/library 编译的 stencil、
rotation feature 等 helper。C 源码和二进制接口说明放在 `tools/cat_surface_c/`；二进制
不入仓库，且不能从本机绝对路径复制到公开仓库。当前 pipeline 是显式 opt-in 的研究后端，
官方 SimNIBS 环境仍保持原实现，完整 CHARM `Surface creation` 尚未在本仓库中宣称替换完成。

## 许可证、上游和作者信息

本仓库中由项目维护者拥有或有权再许可的代码采用 **GNU GPL-3.0-or-later**，完整文本见
`LICENSE`。CAT-Surface 原始代码采用双许可证，且其第三方组件有各自条款；本仓库没有复制
完整 CAT-Surface 源码，`upstream/CAT-Surface-LICENSE.txt` 仅保留上游许可说明。外部依赖
不能被本仓库的许可证声明覆盖，使用者需要自行取得并遵守对应条款。

作者和 AI 辅助贡献说明见 `AUTHORS.md`；上游版本、发布范围和排除项见
`docs/RELEASE_SCOPE.md`。GitHub 官方 `@codex` PR 协作流程见
`docs/CODEX-GITHUB-WORKFLOW.md`。
