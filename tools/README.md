# 工具和 C helper

Python runner 默认从仓库 `src/` 导入包；已安装 editable package 后可以省略
`PYTHONPATH=src`。所有真实输入、输出和 benchmark JSON 都应放在仓库外。

`cat_surface_c/` 下的文件是源码，不是预编译工具。它们需要与目标 CAT-Surface 版本匹配的
headers、静态库/共享库和构建参数；不同系统的 bicpl、FFTW、GIFTI 和 CAT-Surface 链接方式
可能不同，因此本仓库不携带本机编译命令或二进制。构建后通过 Python CLI 的显式参数传入：

- `--reference-cli`：CPU reference CLI；
- `--rotation-values-probe` 或 `--rotation-depth-probe`：官方特征 helper；
- `--stencil-builder`：初始 stencil helper；
- `--rotated-stencil-builder`：旋转后 source stencil helper。

推荐首先运行不需要 C helper 的测试：

```bash
python -m pytest -q
```

涉及真实 GIFTI 的命令必须使用用户自己准备的匿名化/公开 fixture，并在结果目录之外保存
输出。GPU 模式使用 `--device cuda` 时会检查 CUDA 是否真实可见；设备不可用不会静默切换到 CPU。
