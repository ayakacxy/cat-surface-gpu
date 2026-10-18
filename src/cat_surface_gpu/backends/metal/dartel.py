"CAT 正则化双分量膜能系统的 FP32 Metal PCG。"

from functools import lru_cache

import numpy as np

from .surf2sphere import _mlx


_APPLY = r"""
uint i=thread_position_in_grid.x;
uint H=uint(field_shape[1]),W=uint(field_shape[2]),N=H*W;
if(i>=N)return;
uint x=i%W,y=i/W;
uint left=y*W+(x==0?W-1:x-1),right=y*W+(x+1==W?0:x+1);
uint up=(y==0?0:y-1)*W+x,down=(y+1==H?y:y+1)*W+x;
float a=field[i],b=field[N+i],mu=parameters[0],center=4.0f*mu+parameters[1];
output[i]=(center*a-mu*((field[left]+field[right])+field[up]+field[down]))+hessian[i]*a+hessian[2*N+i]*b;
output[N+i]=(center*b-mu*((field[N+left]+field[N+right])+field[N+up]+field[N+down]))+hessian[N+i]*b+hessian[2*N+i]*a;
"""


@lru_cache(maxsize=1)
def _apply_kernel():
    mx = _mlx()
    return mx.fast.metal_kernel(
        name="cat_membrane_apply_fp32",
        input_names=["field", "hessian", "parameters"],
        output_names=["output"],
        source=_APPLY,
        compile_options={"math_mode": "safe"},
    )


def _apply(field, hessian, parameters):
    mx = _mlx()
    return _apply_kernel()(
        inputs=[field, hessian, parameters],
        output_shapes=[field.shape],
        output_dtypes=[mx.float32],
        grid=(field.shape[1] * field.shape[2], 1, 1),
        threadgroup=(128, 1, 1),
        stream=mx.gpu,
    )[0]


def _inverse_blocks(hessian, parameters):
    mx = _mlx()
    diagonal = 4 * parameters[0] + parameters[1]
    a, b, c = hessian[0] + diagonal, hessian[1] + diagonal, hessian[2]
    scale = mx.maximum(mx.maximum(mx.abs(a), mx.abs(b)), mx.maximum(mx.abs(c), 1e-20))
    a, b, c = a / scale, b / scale, c / scale
    denominator = scale * mx.maximum(a * b - c * c, 1e-30)
    return b / denominator, a / denominator, -c / denominator


def _precondition(residual, inverse):
    mx = _mlx()
    a, b, c = inverse
    return mx.stack(
        (a * residual[0] + c * residual[1], b * residual[1] + c * residual[0])
    )


_REDUCE = r"""
threadgroup float scratch[4];
uint lane=thread_position_in_threadgroup.x;
// Apple Metal 的 SIMD 组包含三十二个线程；一百二十八线程块只需一次屏障。
float subtotal=simd_sum(contribution);
if(lane%32==0)scratch[lane/32]=subtotal;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total=simd_sum(lane<4?scratch[lane]:0.0f);
if(lane==0)partial[threadgroup_position_in_grid.x]=total;
"""
_APPLY_REDUCE = (
    r"""
uint i=thread_position_in_grid.x;
uint H=uint(field_shape[1]),W=uint(field_shape[2]),N=H*W;
float contribution=0.0f;
if(i<N){
 uint x=i%W,y=i/W;
 uint left=y*W+(x==0?W-1:x-1),right=y*W+(x+1==W?0:x+1);
 uint up=(y==0?0:y-1)*W+x,down=(y+1==H?y:y+1)*W+x;
 float a=field[i],b=field[N+i],mu=parameters[0],center=4.0f*mu+parameters[1];
 float v0=(center*a-mu*((field[left]+field[right])+field[up]+field[down]))+hessian[i]*a+hessian[2*N+i]*b;
 float v1=(center*b-mu*((field[N+left]+field[N+right])+field[N+up]+field[N+down]))+hessian[N+i]*b+hessian[2*N+i]*a;
 output[i]=v0;output[N+i]=v1;contribution=a*v0+b*v1;
}
"""
    + _REDUCE
)
_UPDATE_REDUCE = (
    r"""
uint i=thread_position_in_grid.x,N=uint(solution_shape[1])*uint(solution_shape[2]);
float contribution=0.0f;
if(i<N){
 float step=scalars[0]/metal::max(scalars[1],1e-30f);
 float r0=residual[i]-step*applied[i],r1=residual[N+i]-step*applied[N+i];
 float z0=inverse[i]*r0+inverse[2*N+i]*r1;
 float z1=inverse[N+i]*r1+inverse[2*N+i]*r0;
 out_solution[i]=solution[i]+step*direction[i];out_solution[N+i]=solution[N+i]+step*direction[N+i];
 out_residual[i]=r0;out_residual[N+i]=r1;out_z[i]=z0;out_z[N+i]=z1;
 contribution=r0*z0+r1*z1;
}
"""
    + _REDUCE
)


@lru_cache(maxsize=1)
def _fused_kernels():
    mx = _mlx()
    apply = mx.fast.metal_kernel(
        name="cat_pcg_apply_reduce",
        input_names=["field", "hessian", "parameters"],
        output_names=["output", "partial"],
        source=_APPLY_REDUCE,
        compile_options={"math_mode": "safe"},
    )
    update = mx.fast.metal_kernel(
        name="cat_pcg_update_reduce",
        input_names=[
            "solution",
            "residual",
            "direction",
            "applied",
            "inverse",
            "scalars",
        ],
        output_names=["out_solution", "out_residual", "out_z", "partial"],
        source=_UPDATE_REDUCE,
        compile_options={"math_mode": "safe"},
    )
    return apply, update


@lru_cache(maxsize=4)
def _fixed_solver(iterations):
    mx = _mlx()
    apply, update = _fused_kernels()

    @mx.compile
    def solve(hessian, rhs, parameters):
        inverse = mx.stack(_inverse_blocks(hessian, parameters))
        solution = mx.zeros_like(rhs)
        residual = rhs
        direction = _precondition(rhs, inverse)
        dot = mx.sum(rhs * direction)
        groups = (rhs.shape[1] * rhs.shape[2] + 127) // 128
        for index in range(iterations):
            # 所有线程均参与部分归约，包括尾部线程。
            applied, partial = apply(
                inputs=[direction, hessian, parameters],
                output_shapes=[rhs.shape, (groups,)],
                output_dtypes=[mx.float32] * 2,
                grid=(groups * 128, 1, 1),
                threadgroup=(128, 1, 1),
                stream=mx.gpu,
            )
            solution, residual, preconditioned, partial = update(
                inputs=[
                    solution,
                    residual,
                    direction,
                    applied,
                    inverse,
                    mx.stack((dot, mx.sum(partial))),
                ],
                output_shapes=[rhs.shape, rhs.shape, rhs.shape, (groups,)],
                output_dtypes=[mx.float32] * 4,
                grid=(groups * 128, 1, 1),
                threadgroup=(128, 1, 1),
                stream=mx.gpu,
            )
            if index + 1 < iterations:
                next_dot = mx.sum(partial)
                beta = mx.where(dot != 0, next_dot / mx.maximum(dot, 1e-30), 0)
                direction = preconditioned + beta * direction
                dot = next_dot
        return solution

    return solve


def solve_membrane_system_metal(
    hessian,
    right_hand_side,
    *,
    mu,
    identity_weight,
    max_iterations=40,
    relative_tolerance=1e-6,
    check_interval=0,
):
    "在 GPU 上执行固定 PCG 迭代，返回一次 CPU FP64 更新。GPU 运算使用 FP32，外围 CAT 形变运算保持 FP64；非零收敛检查间隔保留显式残差检查。"
    mx = _mlx()
    rhs = np.ascontiguousarray(right_hand_side, dtype=np.float32)
    hessian = np.ascontiguousarray(hessian, dtype=np.float32)
    if rhs.ndim != 3 or rhs.shape[0] != 2 or hessian.shape != (3, *rhs.shape[1:]):
        raise ValueError("CAT PCG requires rhs (2,H,W) and Hessian (3,H,W)")
    if not np.isfinite(rhs).all() or not np.isfinite(hessian).all():
        raise ValueError("CAT PCG inputs must be finite")
    if (
        not np.isfinite([mu, identity_weight, relative_tolerance]).all()
        or mu < 0
        or identity_weight <= 0
        or max_iterations < 1
        or relative_tolerance < 0
        or min(rhs.shape[1:]) < 1
    ):
        raise ValueError(
            "CAT Metal PCG requires nonnegative mu, positive damping and iterations"
        )
    if not np.any(rhs):
        return np.zeros(rhs.shape, np.float64), 0
    interval = 1 if check_interval is None else max(0, int(check_interval))
    with mx.stream(mx.gpu):
        h, b = mx.array(hessian), mx.array(rhs)
        parameters = mx.array([mu, identity_weight], mx.float32)
        if interval == 0:
            solution = _fixed_solver(max_iterations)(h, b, parameters)
            completed = max_iterations
        else:
            inverse = _inverse_blocks(h, parameters)
            solution, residual = mx.zeros_like(b), b
            direction = _precondition(residual, inverse)
            dot = mx.sum(residual * direction)
            threshold = relative_tolerance * mx.sqrt(mx.sum(b * b))
            for completed in range(1, max_iterations + 1):
                applied = _apply(direction, h, parameters)
                step = dot / mx.maximum(mx.sum(direction * applied), 1e-30)
                solution, residual = (
                    solution + step * direction,
                    residual - step * applied,
                )
                if completed % interval == 0 or completed == max_iterations:
                    if bool((mx.sqrt(mx.sum(residual * residual)) <= threshold).item()):
                        break
                preconditioned = _precondition(residual, inverse)
                next_dot = mx.sum(residual * preconditioned)
                direction = (
                    preconditioned
                    + mx.where(dot != 0, next_dot / mx.maximum(dot, 1e-30), 0)
                    * direction
                )
                dot = next_dot
                mx.eval(solution, residual, direction, dot)
        mx.eval(solution)
        output = np.array(solution).astype(np.float64)
    if not np.isfinite(output).all():
        raise FloatingPointError("CAT Metal PCG produced a non-finite update")
    return output, completed
