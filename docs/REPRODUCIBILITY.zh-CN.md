# 可复现性与来源记录指南

[English](REPRODUCIBILITY.md) · [简体中文](REPRODUCIBILITY.zh-CN.md)

比较不同运行结果时，应使用相同的输入曲面、模板、参数和求解器。[性能报告](BENCHMARKS_v0.3.0.zh-CN.md)介绍已发布的测量结果，[benchmark-v0.3.0.json](benchmark-v0.3.0.json)保存逐次计时、输入哈希和数值比较结果。

## 记录一次运行

保存运行命令和 JSON 输出，同时记录：

- 软件包版本或 Git 提交；
- 操作系统、Python 版本及后端依赖版本；
- CPU/GPU 型号、CPU 线程数和所选设备；
- 各输入文件的 SHA-256 哈希及顶点、三角面数量；
- 配准参数、求解器和迭代设置。

两条命令均向标准错误输出进度，向标准输出写入最终 JSON 摘要。例如：

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device cuda --threads 8 > sphere.json 2> sphere.log
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device cuda --threads 8 > registration.json 2> registration.log
```

实际设备以 `device` 字段为准。JSON 中的 `backend` 字段是固定的实现名称，不能用它判断该次命令使用了 CPU、CUDA 还是 Metal。

多次运行之间应保持输入文件不变，不同配置的输出分别保存。比较软件版本时，使用不同的源码目录或环境，以便确定实际导入的软件包和编译缓存。

## 比较曲面

先检查顶点数、三角面是否完全一致、坐标是否有限，以及是否存在零面积三角形，再比较对应顶点。常用指标包括欧氏距离的均值、最大值和第 95 百分位数。球面输出还可以用角度差描述，单位为度。

报告结果时应注明指标和单位：逐坐标绝对误差与逐顶点欧氏距离不是同一个量。坐标一致性用于衡量数值复现情况；解剖配准质量需要通过标志点或区域对应等解剖指标评估。

算法参考版本固定在 [UPSTREAM.lock](../UPSTREAM.lock) 中。参考实现使用全多重网格法（FMG），本项目默认使用预条件共轭梯度法（PCG）；CPU 和 CUDA 也提供 FMG 选项。比较时应记录实际求解器，仅使用相同输入并不能使两种数值方法等价。

## 测量耗时

完整命令的计时从进程启动前开始，到进程退出后结束，包括 Python 启动、模块导入、文件读取、后端初始化、计算和文件写出。CLI 的 `wall_seconds` 从参数解析完成后开始；API 的阶段计时覆盖范围更小。报告时应明确区分这三种计时范围。

首次运行与缓存已建立后的耗时分别报告。首次运行可能包含 Numba/CuPy/MLX 编译以及拓扑、模板准备。重复运行时，说明是使用已有磁盘缓存启动新进程，还是在同一个 Python 进程中重复调用。至少测量三次，并报告中位数和范围。

双半球任务需注明计时覆盖一个还是两个半球、串行还是并发执行，以及是否等到所有输出文件写完。手动计时时，应在停止计时前同步 GPU 工作。

仓库提供 `tools/benchmark_versions.py`，用于比较两份源码的完整命令耗时。它记录首次和重复运行耗时、源码与输入哈希，以及输出差异。所需输入路径和选项可通过以下命令查看：

```bash
python tools/benchmark_versions.py --help
```

## 缓存与测试数据

`CAT_SURFACE_CACHE_ROOT` 控制本项目的拓扑与模板缓存，默认位于当前工作目录下的 `.work/cat-surface/cache`。Numba 和 CuPy 分别通过 `NUMBA_CACHE_DIR` 和 `CUPY_CACHE_DIR` 设置编译缓存。测量首次运行耗时时，可使用独立缓存目录。[构建指南](BUILDING.zh-CN.md#准备环境)提供了 Linux、macOS 和 Windows 的项目内缓存设置示例。

合成网格适合安装、文件读写和小规模数值测试。完整球面生成与配准应使用能够代表目标解剖流程的网格：过粗的网格可能在膨胀过程中塌缩，其耗时也不能代表皮层网格。仓库不分发被试数据或解剖模板。

[文档目录](README.zh-CN.md) · [项目首页](../README.zh-CN.md)
