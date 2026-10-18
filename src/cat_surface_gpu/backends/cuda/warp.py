"使用局部位移插值的 GPU 常驻 FP32 CUDA DARTEL。"

from functools import lru_cache
import numpy as np
from cat_surface_gpu.backends.cuda.pcg_fp32 import solve_fixed_pcg, _get_pcg_kernels

_SOURCE = r"""

__device__ inline int wrap(int x,int w){int z=x%w;return z<0?z+w:z;}
__device__ inline int reflect(int y,int h){int z=wrap(y,2*h);return z>=h?2*h-z-1:z;}
struct Stencil {int a,b,c,d,x0,x1,y0,y1;float fx,fy;};
__device__ inline Stencil stencil(const float* displacement,int i,int n,int w,int h){
 float dx=displacement[i],dy=displacement[n+i];
 int ix=int(floor(dx)),iy=int(floor(dy));
 Stencil s;s.fx=dx-float(ix);s.fy=dy-float(iy);
 s.x0=wrap(i%w+ix,w);s.x1=wrap(i%w+ix+1,w);
 s.y0=reflect(i/w+iy,h);s.y1=reflect(i/w+iy+1,h);
 s.a=s.y0*w+s.x0;s.b=s.y0*w+s.x1;s.c=s.y1*w+s.x0;s.d=s.y1*w+s.x1;return s;
}
__device__ inline float sample(const float* f,Stencil s){
 return (f[s.a]*(1-s.fx)+f[s.b]*s.fx)*(1-s.fy)+(f[s.c]*(1-s.fx)+f[s.d]*s.fx)*s.fy;
}
__device__ inline float map_sample(const float* f,Stencil s,int i,int w,int h,int axis){
 // 先插值局部位移再加坐标，避免绝对坐标吞掉微小更新。
 float base=axis==0?float(s.x0-i%w):float(s.y0-i/w);
 float b=axis==0?float(s.x1-s.x0):0.0f;
 float c=axis==0?0.0f:float(s.y1-s.y0);
 float d=b+c,p=axis==0?float(w):float(h);
 b+=f[s.b]-f[s.a];c+=f[s.c]-f[s.a];d+=f[s.d]-f[s.a];
 b-=floor(b/p+0.5f)*p;c-=floor(c/p+0.5f)*p;d-=floor(d/p+0.5f)*p;
 float value=base+f[s.a]+(b*s.fx)*(1-s.fy)+(c*(1-s.fx)+d*s.fx)*s.fy;
 if(axis==0)value-=floor(value/p+0.5f)*p;
 return value;
}
__device__ inline void compose_at(const float* displacement,const float* jacobian,
 float* out_displacement,float* out_jacobian,Stencil s,int i,int n,int w,int h){
 out_displacement[i]=map_sample(displacement,s,i,w,h,0);
 out_displacement[n+i]=map_sample(displacement+n,s,i,w,h,1);
 float b00=sample(jacobian,s),b01=sample(jacobian+n,s),b10=sample(jacobian+2*n,s),b11=sample(jacobian+3*n,s);
 float a00=jacobian[i],a01=jacobian[n+i],a10=jacobian[2*n+i],a11=jacobian[3*n+i];
 out_jacobian[i]=a00*b00+a01*b10;out_jacobian[n+i]=a00*b01+a01*b11;
 out_jacobian[2*n+i]=a10*b00+a11*b10;out_jacobian[3*n+i]=a10*b01+a11*b11;
}

extern "C" __global__ void small(const float* velocity,const float* scale,float* displacement,float* jacobian,int h,int w){

int i=blockIdx.x*blockDim.x+threadIdx.x,n=h*w;
if(i>=n)return;
int x=i%w,y=i/w,l=y*w+wrap(x-1,w),r=y*w+wrap(x+1,w);
int u=reflect(y-1,h)*w+x,d=reflect(y+1,h)*w+x;float s=scale[0],a=0.5f*s;
displacement[i]=velocity[i]*s;displacement[n+i]=velocity[n+i]*s;
jacobian[i]=1+a*(velocity[r]-velocity[l]);jacobian[n+i]=a*(velocity[n+r]-velocity[n+l]);
jacobian[2*n+i]=a*(velocity[d]-velocity[u]);jacobian[3*n+i]=1+a*(velocity[n+d]-velocity[n+u]);

}

extern "C" __global__ void compose(const float* displacement,const float* jacobian,float* out_displacement,float* out_jacobian,int h,int w){

int i=blockIdx.x*blockDim.x+threadIdx.x,n=h*w;
if(i>=n)return;Stencil s=stencil(displacement,i,n,w,h);
compose_at(displacement,jacobian,out_displacement,out_jacobian,s,i,n,w,h);

}

extern "C" __global__ void objective(const float* source,const float* target,const float* displacement,const float* jacobian,const float* small_jacobian,float* gradient,float* hessian,float* errors,int h,int w){

int i=blockIdx.x*blockDim.x+threadIdx.x,n=h*w;
if(i>=n)return;Stencil s=stencil(displacement,i,n,w,h);
float a=small_jacobian[i],b=small_jacobian[n+i],c=small_jacobian[2*n+i],d=small_jacobian[3*n+i];
float inv=1.0f/(a*d-b*c),i00=d*inv,i01=-c*inv,i10=-b*inv,i11=a*inv;
float q00=jacobian[i],q01=jacobian[2*n+i],q10=jacobian[n+i],q11=jacobian[3*n+i];
float r00=q00*i00+q01*i10,r01=q10*i00+q11*i10,r10=q00*i01+q01*i11,r11=q10*i01+q11*i11;
float k22=source[s.a],k12=source[s.b],k21=source[s.c],k11=source[s.d],fx=s.fx,fy=s.fy;
float top=k11*fx+k21*(1-fx),bottom=k12*fx+k22*(1-fx);
float delta=top*fy+bottom*(1-fy)-target[i];
float dx=(k11-k21)*fy+(k12-k22)*(1-fy),dy=top-bottom;
float gx=r00*dx+r01*dy,gy=r10*dx+r11*dy;
gradient[i]=gx*delta;gradient[n+i]=gy*delta;
hessian[i]=gx*gx;hessian[n+i]=gy*gy;hessian[2*n+i]=gx*gy;errors[i]=0.5f*delta*delta;

}

extern "C" __global__ void transport(const float* gradient,const float* hessian,const float* displacement,const float* jacobian,float* out_gradient,float* out_hessian,float* out_displacement,float* out_jacobian,int h,int w){

int i=blockIdx.x*blockDim.x+threadIdx.x,n=h*w;
if(i>=n)return;Stencil s=stencil(displacement,i,n,w,h);
float first=sample(gradient,s),second=sample(gradient+n,s);
float a00=sample(hessian,s),a11=sample(hessian+n,s),a01=sample(hessian+2*n,s);
float j11=jacobian[i],j21=jacobian[n+i],j12=jacobian[2*n+i],j22=jacobian[3*n+i];
float det=j11*j22-j12*j21;
out_gradient[i]=gradient[i]+det*(first*j11+second*j21);
out_gradient[n+i]=gradient[n+i]+det*(first*j12+second*j22);
float t1=a00*j11+a01*j21,t2=a01*j11+a11*j21,t3=a00*j12+a01*j22,t4=a01*j12+a11*j22;
out_hessian[i]=hessian[i]+det*(t1*j11+t2*j21);
out_hessian[n+i]=hessian[n+i]+det*(t3*j12+t4*j22);
out_hessian[2*n+i]=hessian[2*n+i]+det*(t1*j12+t2*j22);
compose_at(displacement,jacobian,out_displacement,out_jacobian,s,i,n,w,h);

}
"""


@lru_cache(maxsize=8)
def _kernels(device):
    "按设备缓存四组融合核。"
    import cupy as cp

    module = cp.RawModule(code=_SOURCE, options=("--std=c++11", "--fmad=false"))
    return {
        name: module.get_function(name)
        for name in ("small", "compose", "objective", "transport")
    }


def _call(name, inputs, shapes):
    "按空间形状分配梯度、Hessian 或形变输出并启动内核。"
    import cupy as cp

    outputs = tuple(cp.empty(shape, cp.float32) for shape in shapes)
    h, w = shapes[0][-2:]
    _kernels(cp.cuda.Device().id)[name](
        ((h * w + 127) // 128,), (128,), (*inputs, *outputs, np.int32(h), np.int32(w))
    )
    return outputs


def small(velocity, scale):
    "构建小形变位移和雅可比。"
    import cupy as cp

    h, w = velocity.shape[1:]
    return _call(
        "small", (velocity, cp.asarray([scale], cp.float32)), ((2, h, w), (2, 2, h, w))
    )


def compose(displacement, jacobian):
    "复合局部位移及其雅可比。"
    return _call(
        "compose", (displacement, jacobian), (displacement.shape, jacobian.shape)
    )


def objective(source, target, displacement, jacobian, small_jacobian):
    "计算对称目标函数的梯度、Hessian 和误差。"
    h, w = source.shape
    return _call(
        "objective",
        (source, target, displacement, jacobian, small_jacobian),
        ((2, h, w), (3, h, w), (h, w)),
    )


def transport(gradient, hessian, displacement, jacobian, squarings):
    "通过重复平方运输导数。"
    for _ in range(squarings):
        gradient, hessian, displacement, jacobian = _call(
            "transport",
            (gradient, hessian, displacement, jacobian),
            (gradient.shape, hessian.shape, displacement.shape, jacobian.shape),
        )
    return gradient, hessian, displacement, jacobian


def step(velocity, source, target, squarings, iterations, mu, identity_weight):
    "执行对称形变更新，默认保留四十次 PCG 迭代。"
    import cupy as cp

    scale = 2.0**-squarings
    forward, forward_jacobian = small(velocity, scale)
    mapping, jacobian = forward, forward_jacobian
    for _ in range(squarings):
        mapping, jacobian = compose(mapping, jacobian)
    gradient, hessian, error = objective(
        source, target, mapping, jacobian, forward_jacobian
    )
    backward, backward_jacobian = small(velocity, -scale)
    gradient, hessian, mapping, jacobian = transport(
        gradient, hessian, backward, backward_jacobian, squarings
    )
    reverse_gradient, reverse_hessian, reverse_error = objective(
        target, source, mapping, jacobian, backward_jacobian
    )
    reverse_gradient, reverse_hessian, _, _ = transport(
        reverse_gradient, reverse_hessian, forward, forward_jacobian, squarings
    )
    h, w = source.shape
    regularization = cp.empty_like(velocity)
    zeros = cp.zeros_like(hessian)
    _get_pcg_kernels(cp.cuda.Device().id)[0](
        ((h * w + 255) // 256,),
        (256,),
        (
            velocity[0],
            velocity[1],
            *zeros,
            regularization[0],
            regularization[1],
            np.int32(h),
            np.int32(w),
            np.float32(mu),
            np.float32(0),
        ),
    )
    rhs = (gradient - reverse_gradient) * scale + regularization
    update, _ = solve_fixed_pcg(
        (hessian + reverse_hessian) * scale, rhs, mu, identity_weight, iterations
    )
    return velocity - update, cp.sum(error) + cp.sum(reverse_error)


def run_schedule(source, target, schedule, iterations=40):
    "将尺度调度保持在设备端，并恢复最终 FP64 指数映射接口。"
    import cupy as cp

    source = np.ascontiguousarray(source, np.float32)
    target = np.ascontiguousarray(target, np.float32)
    if (
        source.ndim != 2
        or source.shape != target.shape
        or min(source.shape) < 1
        or iterations < 1
    ):
        raise ValueError("Invalid CUDA DARTEL sheets or iterations")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("CUDA DARTEL sheets must be finite")
    first, second = cp.asarray(source), cp.asarray(target)
    velocity = cp.zeros((2, *source.shape), cp.float32)
    objectives = []
    for item in schedule:
        for _ in range(int(item["iterations"])):
            velocity, value = step(
                velocity,
                first,
                second,
                int(item["time_step"]),
                iterations,
                float(item["mu"]),
                float(item["lmreg"]),
            )
            objectives.append(value)
    values = cp.stack(objectives) if objectives else cp.empty(0, cp.float32)
    if not bool(cp.isfinite(velocity).all().item()) or not bool(
        cp.isfinite(values).all().item()
    ):
        raise FloatingPointError("CAT resident CUDA DARTEL produced non-finite outputs")
    return velocity.astype(cp.float64), values.astype(cp.float64)
