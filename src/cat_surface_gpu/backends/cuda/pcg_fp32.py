"使用固定工作区将 FP32 PCG 捕获到 CUDA 计算图。"

from functools import lru_cache
import numpy as np

_SOURCE = r"""

            extern "C" __global__ void apply_membrane_hessian(
                const float* field0, const float* field1,
                const float* hessian0, const float* hessian1,
                const float* hessian2, float* output0, float* output1,
                int height, int width, float mu, float identity_weight) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                int y = index / width;
                int x = index - y * width;
                int left_x = (x == 0) ? width - 1 : x - 1;
                int right_x = (x + 1 == width) ? 0 : x + 1;
                int up_y = (y == 0) ? 0 : y - 1;
                int down_y = (y + 1 == height) ? height - 1 : y + 1;
                int left = y * width + left_x;
                int right = y * width + right_x;
                int up = up_y * width + x;
                int down = down_y * width + x;
                float center = 4.0f * mu + identity_weight;
                float membrane0 = center * field0[index]
                    - mu * (field0[left] + field0[right]
                            + field0[up] + field0[down]);
                float membrane1 = center * field1[index]
                    - mu * (field1[left] + field1[right]
                            + field1[up] + field1[down]);
                output0[index] = membrane0
                    + hessian0[index] * field0[index]
                    + hessian2[index] * field1[index];
                output1[index] = membrane1
                    + hessian1[index] * field1[index]
                    + hessian2[index] * field0[index];
            }

            extern "C" __global__ void apply_preconditioner(
                const float* residual0, const float* residual1,
                const float* hessian0, const float* hessian1,
                const float* hessian2, float* output0, float* output1,
                int count, float diagonal) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                float diagonal0 = hessian0[index] + diagonal;
                float diagonal1 = hessian1[index] + diagonal;
                float cross = hessian2[index];
                float scale=fmaxf(fmaxf(fabsf(diagonal0),fabsf(diagonal1)),fmaxf(fabsf(cross),1e-20f));
                diagonal0/=scale;diagonal1/=scale;cross/=scale;
                float denominator=scale*fmaxf(diagonal0*diagonal1-cross*cross,1e-30f);
                output0[index]=(diagonal1/denominator)*residual0[index]-(cross/denominator)*residual1[index];
                output1[index]=(diagonal0/denominator)*residual1[index]-(cross/denominator)*residual0[index];
            }

            extern "C" __global__ void update_solution_residual(
                float* solution0, float* solution1,
                float* residual0, float* residual1,
                const float* direction0, const float* direction1,
                const float* applied0, const float* applied1,
                const float* step, int count) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                float value = *step;
                solution0[index] += value * direction0[index];
                solution1[index] += value * direction1[index];
                residual0[index] -= value * applied0[index];
                residual1[index] -= value * applied1[index];
            }

            extern "C" __global__ void update_direction(
                float* direction0, float* direction1,
                const float* preconditioned0,
                const float* preconditioned1,
                const float* beta, int count) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                float value = *beta;
                direction0[index] = preconditioned0[index]
                    + value * direction0[index];
                direction1[index] = preconditioned1[index]
                    + value * direction1[index];
            }
            """


@lru_cache(maxsize=8)
def _get_pcg_kernels(device):
    "按设备缓存矩阵、预条件器和向量更新核。"
    import cupy as cp

    module = cp.RawModule(code=_SOURCE, options=("--std=c++11", "--fmad=false"))
    return tuple(
        module.get_function(name)
        for name in (
            "apply_membrane_hessian",
            "apply_preconditioner",
            "update_solution_residual",
            "update_direction",
        )
    )


class FixedPCG:
    "使用私有可变输入缓冲，返回与复用工作区独立的结果。"

    def __init__(self, shape, mu, identity_weight, iterations):
        import cupy as cp

        self.cp = cp
        self.shape = shape
        self.height, self.width = shape[1:]
        self.count = self.height * self.width
        self.blocks = (self.count + 255) // 256
        self.mu = float(mu)
        self.identity_weight = float(identity_weight)
        self.diagonal = 4.0 * self.mu + self.identity_weight
        self.iterations = int(iterations)
        self.kernels = _get_pcg_kernels(cp.cuda.Device().id)
        self.hessian = cp.empty((3, self.height, self.width), dtype=cp.float32)
        self.rhs = cp.empty(shape, dtype=cp.float32)
        self.solution = cp.empty_like(self.rhs)
        self.residual = cp.empty_like(self.rhs)
        self.preconditioned = cp.empty_like(self.rhs)
        self.direction = cp.empty_like(self.rhs)
        self.applied = cp.empty_like(self.rhs)
        self.product = cp.empty_like(self.rhs)
        self.dot, self.next_dot, self.denominator, self.step, self.beta = (
            cp.empty((), dtype=cp.float32) for _ in range(5)
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
                np.float32(self.diagonal),
            ),
        )

    def _dot(self, left, right, output):
        self.cp.multiply(left, right, out=self.product)
        self.cp.sum(self.product, dtype=self.cp.float32, out=output)

    def _launch(self):
        "保留原始乘法、求和、标量除法及 PCG 更新顺序。"
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
                    np.float32(self.mu),
                    np.float32(self.identity_weight),
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
            cp.maximum(self.dot, 1e-30, out=self.beta)
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
        "记录固定依赖，每次重放刷新输入并返回独立结果。"
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
    "按设备、网格和原始求解参数限制缓存数量。"
    return FixedPCG(shape, mu, identity_weight, iterations)


def solve_fixed_pcg(hessian, rhs, mu, identity_weight, iterations):
    "仅处理无中途收敛检查的固定迭代 CUDA 求解。"
    import cupy as cp

    return _workspace(
        cp.cuda.Device().id,
        cp.cuda.get_current_stream().ptr,
        tuple(rhs.shape),
        float(mu),
        float(identity_weight),
        int(iterations),
    ).solve(hessian, rhs)
