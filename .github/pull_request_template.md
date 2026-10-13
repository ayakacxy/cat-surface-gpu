## Summary / 修改说明

Describe the change and its scientific or engineering motivation.

说明修改内容及其科学或工程目的。

## Validation / 验证

- [ ] CPU tests pass. / CPU 测试通过。
- [ ] CUDA tests were run on a real GPU, or the PR clearly states that they were not run. / 已在真实 GPU 上运行 CUDA 测试，或在 PR 中明确说明未运行。
- [ ] Reference and optimized paths use identical inputs, parameters, shapes, and precision. / 参考实现与优化实现使用相同输入、参数、数组形状和精度。
- [ ] Topology and max/mean/p99 numerical errors are reported for algorithm changes. / 算法修改已报告拓扑差异及最大、平均和第 99 百分位数值误差。
- [ ] Performance claims include hardware, synchronization, repetitions, and timing boundaries. / 性能结果已说明硬件、同步方式、重复次数和计时范围。
- [ ] No resolution, iteration, stopping condition, reduction order, or tie-breaking semantics changed. / 未改变分辨率、迭代次数、停止条件、归约顺序或相同值的选择规则。
- [ ] No identifiable medical data, build cache, or generated binary was added unintentionally. / 未意外加入可识别身份的医学数据、构建缓存或生成的二进制文件。
- [ ] New code comments and docstrings are in Chinese. / 新增代码注释和 docstring 使用中文。

## License and provenance / 许可与来源

- [ ] New dependencies and copied code are documented in `THIRD_PARTY_NOTICES.md`. / 新增依赖和复制的代码已记录在 `THIRD_PARTY_NOTICES.md` 中。
- [ ] The package runs without CAT executables or project-specific native extensions. / 软件包不依赖 CAT 可执行程序或本项目专用的原生扩展即可运行。
