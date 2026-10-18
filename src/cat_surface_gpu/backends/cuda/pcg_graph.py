"固定迭代 PCG 的设备缓冲和计算图，保留原归约及更新核。"

from functools import lru_cache
import numpy as np


class FixedPCG:
    "可变输入使用独立缓冲，返回与复用工作区独立的结果。"

    def __init__(self, shape, mu, identity_weight, iterations):
        import cupy as cp
        from cat_surface_gpu.dartel import _get_pcg_kernels

        self.cp = cp
        self.shape = shape
        self.height, self.width = shape[1:]
        self.count = self.height * self.width
        self.blocks = (self.count + 255) // 256
        self.mu = float(mu)
        self.identity_weight = float(identity_weight)
        self.diagonal = 4.0 * self.mu + self.identity_weight
        self.iterations = int(iterations)
        self.kernels = _get_pcg_kernels()
        self.hessian = cp.empty((3, self.height, self.width), dtype=cp.float64)
        self.rhs = cp.empty(shape, dtype=cp.float64)
        self.solution = cp.empty_like(self.rhs)
        self.residual = cp.empty_like(self.rhs)
        self.preconditioned = cp.empty_like(self.rhs)
        self.direction = cp.empty_like(self.rhs)
        self.applied = cp.empty_like(self.rhs)
        self.product = cp.empty_like(self.rhs)
        self.dot, self.next_dot, self.denominator, self.step, self.beta = (
            cp.empty((), dtype=cp.float64) for _ in range(5)
        )
        # 归约临时内存由本图独占，避免内存池复用。
        self.pool = cp.cuda.MemoryPool()
        self.graph = None

    def _precondition(self):
        self.kernels[1](
            (self.blocks,),
            (256,),
            (
                self.residual[0],
                self.residual[1],
                *self.hessian,
                self.preconditioned[0],
                self.preconditioned[1],
                np.int32(self.count),
                np.float64(self.diagonal),
            ),
        )

    def _dot(self, left, right, output):
        self.cp.multiply(left, right, out=self.product)
        self.cp.sum(self.product, dtype=self.cp.float64, out=output)

    def _launch(self):
        "保留乘法、求和、标量除法及 PCG 更新顺序。"
        cp = self.cp
        self.solution.fill(0)
        cp.copyto(self.residual, self.rhs)
        self._precondition()
        cp.copyto(self.direction, self.preconditioned)
        self._dot(self.residual, self.preconditioned, self.dot)
        for completed in range(1, self.iterations + 1):
            self.kernels[0](
                (self.blocks,),
                (256,),
                (
                    self.direction[0],
                    self.direction[1],
                    *self.hessian,
                    self.applied[0],
                    self.applied[1],
                    np.int32(self.height),
                    np.int32(self.width),
                    np.float64(self.mu),
                    np.float64(self.identity_weight),
                ),
            )
            self._dot(self.direction, self.applied, self.denominator)
            cp.maximum(self.denominator, 1e-30, out=self.denominator)
            cp.divide(self.dot, self.denominator, out=self.step)
            self.kernels[2](
                (self.blocks,),
                (256,),
                (
                    self.solution[0],
                    self.solution[1],
                    self.residual[0],
                    self.residual[1],
                    self.direction[0],
                    self.direction[1],
                    self.applied[0],
                    self.applied[1],
                    self.step,
                    np.int32(self.count),
                ),
            )
            if completed == self.iterations:
                break
            self._precondition()
            self._dot(self.residual, self.preconditioned, self.next_dot)
            cp.copyto(self.beta, cp.where(self.dot == 0.0, 1.0, self.dot))
            cp.divide(self.next_dot, self.beta, out=self.beta)
            self.kernels[3](
                (self.blocks,),
                (256,),
                (
                    self.direction[0],
                    self.direction[1],
                    self.preconditioned[0],
                    self.preconditioned[1],
                    self.beta,
                    np.int32(self.count),
                ),
            )
            cp.copyto(self.dot, self.next_dot)

    def solve(self, hessian, rhs):
        "捕获固定依赖，每次刷新输入并返回独立结果数组。"
        cp = self.cp
        cp.copyto(self.hessian, hessian)
        cp.copyto(self.rhs, rhs)
        if self.graph is None:
            stream = cp.cuda.Stream(non_blocking=True)
            ready = cp.cuda.get_current_stream().record()
            with stream, cp.cuda.using_allocator(self.pool.malloc):
                stream.wait_event(ready)
                self._launch()
                stream.synchronize()
                stream.begin_capture()
                self._launch()
                self.graph = stream.end_capture()
            self.stream = stream
        self.graph.launch(cp.cuda.get_current_stream())
        return self.solution.copy(), self.iterations


@lru_cache(maxsize=8)
def _workspace(device, stream, shape, mu, identity_weight, iterations):
    "按设备、网格和求解参数限制工作区缓存数量。"
    return FixedPCG(shape, mu, identity_weight, iterations)


def solve_fixed_pcg(hessian, rhs, mu, identity_weight, iterations):
    "仅执行固定迭代预算且无中途收敛检查的 CUDA PCG。"
    import cupy as cp

    return _workspace(
        cp.cuda.Device().id,
        cp.cuda.get_current_stream().ptr,
        tuple(rhs.shape),
        float(mu),
        float(identity_weight),
        int(iterations),
    ).solve(hessian, rhs)
