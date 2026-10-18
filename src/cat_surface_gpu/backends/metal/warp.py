"使用位移插值的常驻 FP32 球面 DARTEL。"

from functools import lru_cache
import numpy as np
from .surf2sphere import _mlx
from .dartel import _fixed_solver, _apply

_HEADER = r"""
inline int wrap(int x,int w){int z=x%w;return z<0?z+w:z;}
inline int reflect(int y,int h){int z=wrap(y,2*h);return z>=h?2*h-z-1:z;}
struct Stencil {int a,b,c,d,x0,x1,y0,y1;float fx,fy;};
inline Stencil stencil(device const float* displacement,int i,int n,int w,int h){
 float dx=displacement[i],dy=displacement[n+i];
 int ix=int(floor(dx)),iy=int(floor(dy));
 Stencil s;s.fx=dx-float(ix);s.fy=dy-float(iy);
 s.x0=wrap(i%w+ix,w);s.x1=wrap(i%w+ix+1,w);
 s.y0=reflect(i/w+iy,h);s.y1=reflect(i/w+iy+1,h);
 s.a=s.y0*w+s.x0;s.b=s.y0*w+s.x1;s.c=s.y1*w+s.x0;s.d=s.y1*w+s.x1;return s;
}
inline float sample(device const float* f,Stencil s){
 return (f[s.a]*(1-s.fx)+f[s.b]*s.fx)*(1-s.fy)+(f[s.c]*(1-s.fx)+f[s.d]*s.fx)*s.fy;
}
inline float map_sample(device const float* f,Stencil s,int i,int w,int h,int axis){
 // 插值相对锚点的偏移，避免大网格坐标吞掉亚像素位移。
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
inline void compose_at(device const float* displacement,device const float* jacobian,
 device float* out_displacement,device float* out_jacobian,Stencil s,int i,int n,int w,int h){
 out_displacement[i]=map_sample(displacement,s,i,w,h,0);
 out_displacement[n+i]=map_sample(displacement+n,s,i,w,h,1);
 float b00=sample(jacobian,s),b01=sample(jacobian+n,s),b10=sample(jacobian+2*n,s),b11=sample(jacobian+3*n,s);
 float a00=jacobian[i],a01=jacobian[n+i],a10=jacobian[2*n+i],a11=jacobian[3*n+i];
 out_jacobian[i]=a00*b00+a01*b10;out_jacobian[n+i]=a00*b01+a01*b11;
 out_jacobian[2*n+i]=a10*b00+a11*b10;out_jacobian[3*n+i]=a10*b01+a11*b11;
}
"""
_SMALL = r"""
int i=int(thread_position_in_grid.x),h=int(velocity_shape[1]),w=int(velocity_shape[2]),n=h*w;
if(i>=n)return;
int x=i%w,y=i/w,l=y*w+wrap(x-1,w),r=y*w+wrap(x+1,w);
int u=reflect(y-1,h)*w+x,d=reflect(y+1,h)*w+x;float s=scale[0],a=0.5f*s;
displacement[i]=velocity[i]*s;displacement[n+i]=velocity[n+i]*s;
jacobian[i]=1+a*(velocity[r]-velocity[l]);jacobian[n+i]=a*(velocity[n+r]-velocity[n+l]);
jacobian[2*n+i]=a*(velocity[d]-velocity[u]);jacobian[3*n+i]=1+a*(velocity[n+d]-velocity[n+u]);
"""
_COMPOSE = r"""
int i=int(thread_position_in_grid.x),h=int(displacement_shape[1]),w=int(displacement_shape[2]),n=h*w;
if(i>=n)return;Stencil s=stencil(displacement,i,n,w,h);
compose_at(displacement,jacobian,out_displacement,out_jacobian,s,i,n,w,h);
"""
_OBJECTIVE = r"""
int i=int(thread_position_in_grid.x),h=int(source_shape[0]),w=int(source_shape[1]),n=h*w;
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
"""
_TRANSPORT = r"""
int i=int(thread_position_in_grid.x),h=int(displacement_shape[1]),w=int(displacement_shape[2]),n=h*w;
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
"""


@lru_cache(maxsize=1)
def _kernels():
    mx = _mlx()
    definitions = {
        "small": (["velocity", "scale"], ["displacement", "jacobian"], _SMALL),
        "compose": (
            ["displacement", "jacobian"],
            ["out_displacement", "out_jacobian"],
            _COMPOSE,
        ),
        "objective": (
            ["source", "target", "displacement", "jacobian", "small_jacobian"],
            ["gradient", "hessian", "errors"],
            _OBJECTIVE,
        ),
        "transport": (
            ["gradient", "hessian", "displacement", "jacobian"],
            ["out_gradient", "out_hessian", "out_displacement", "out_jacobian"],
            _TRANSPORT,
        ),
    }
    return {
        name: mx.fast.metal_kernel(
            name="cat_warp_" + name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_HEADER,
            compile_options={"math_mode": "safe"},
        )
        for name, (inputs, outputs, source) in definitions.items()
    }


def _call(name, inputs, shapes):
    mx = _mlx()
    h, w = shapes[0][-2:]
    return _kernels()[name](
        inputs=inputs,
        output_shapes=shapes,
        output_dtypes=[mx.float32] * len(shapes),
        grid=(h * w, 1, 1),
        threadgroup=(128, 1, 1),
        stream=mx.gpu,
    )


def small(velocity, scale):
    mx = _mlx()
    h, w = velocity.shape[1:]
    return _call(
        "small", [velocity, mx.array([scale], mx.float32)], [(2, h, w), (2, 2, h, w)]
    )


def compose(displacement, jacobian):
    return _call(
        "compose", [displacement, jacobian], [displacement.shape, jacobian.shape]
    )


def exponentiate(velocity, squarings):
    displacement, jacobian = small(velocity, 2.0**-squarings)
    for _ in range(squarings):
        displacement, jacobian = compose(displacement, jacobian)
    return displacement, jacobian


def objective(source, target, displacement, jacobian, small_jacobian):
    h, w = source.shape
    return _call(
        "objective",
        [source, target, displacement, jacobian, small_jacobian],
        [(2, h, w), (3, h, w), (h, w)],
    )


def transport(gradient, hessian, displacement, jacobian, squarings):
    for _ in range(squarings):
        gradient, hessian, displacement, jacobian = _call(
            "transport",
            [gradient, hessian, displacement, jacobian],
            [gradient.shape, hessian.shape, displacement.shape, jacobian.shape],
        )
    return gradient, hessian, displacement, jacobian


@lru_cache(maxsize=12)
def _step(squarings, iterations):
    mx = _mlx()
    solve = _fixed_solver(iterations)

    @mx.compile
    def run(velocity, source, target, parameters):
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
        # Hessian 为零时，既有应用核仅保留膜能模板。
        regularization = _apply(
            velocity,
            mx.zeros_like(hessian),
            mx.stack((parameters[0], mx.array(0, mx.float32))),
        )
        rhs = (gradient - reverse_gradient) * scale + regularization
        update = solve((hessian + reverse_hessian) * scale, rhs, parameters)
        return velocity - update, mx.sum(error) + mx.sum(reverse_error)

    return run


def run_schedule(source, target, schedule, iterations=40):
    "仅上传一次特征网格，将全部 DARTEL 状态保留在 GPU。"
    mx = _mlx()
    source = np.ascontiguousarray(source, np.float32)
    target = np.ascontiguousarray(target, np.float32)
    if source.ndim != 2 or source.shape != target.shape or min(source.shape) < 1:
        raise ValueError("CAT feature sheets must have matching nonempty 2D shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("CAT feature sheets must be finite")
    with mx.stream(mx.gpu):
        first, second = mx.array(source), mx.array(target)
        velocity = mx.zeros((2, *source.shape), mx.float32)
        objectives = []
        for item in schedule:
            runner = _step(int(item["time_step"]), iterations)
            parameters = mx.array([item["mu"], item["lmreg"]], mx.float32)
            for _ in range(int(item["iterations"])):
                velocity, value = runner(velocity, first, second, parameters)
                objectives.append(value)
            # 将惰性图限制在一个调度项内，不向主机提取标量。
            mx.eval(velocity, objectives)
        mx.eval(velocity)
        result = np.array(velocity).astype(np.float64)
        values = np.array(mx.stack(objectives)).astype(np.float64)
    if not np.isfinite(result).all() or not np.isfinite(values).all():
        raise FloatingPointError(
            "CAT resident Metal DARTEL produced non-finite outputs"
        )
    return result, values
