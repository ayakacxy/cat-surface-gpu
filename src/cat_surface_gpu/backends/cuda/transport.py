"融合 DARTEL 五通道导数采样和雅可比累加。"

from functools import lru_cache
import numpy as np

SOURCE = r"""
// 按 FP64 顺序计算周期经度、反射纬度和双线性权重。
__device__ int bound_x(long long x,int width) {
    long long value=x%width;return (int)(value<0?value+width:value);
}
__device__ int bound_y(long long y,int height) {
    if(height==1)return 0;
    long long period=2LL*height,value=y%period;
    if(value<0)value+=period;
    return (int)(value>=height?period-value-1:value);
}
__device__ double sample(const double* field,int a,int b,int c,int d,
    double fx,double fy,double ix,double iy) {
    return (field[a]*ix+field[b]*fx)*iy+(field[c]*ix+field[d]*fx)*fy;
}
extern "C" __global__ void transport_accumulate(const double* gradient,
    const double* hessian,const double* mapping,const double* jacobian,
    int height,int width,double* output_gradient,double* output_hessian) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,n=height*width;
    if(i>=n)return;
    double x=mapping[i]-1.0,y=mapping[n+i]-1.0;
    long long floor_x=(long long)floor(x),floor_y=(long long)floor(y);
    double fx=x-floor_x,fy=y-floor_y,ix=1.0-fx,iy=1.0-fy;
    int x0=bound_x(floor_x,width),x1=bound_x(floor_x+1,width);
    int y0=bound_y(floor_y,height),y1=bound_y(floor_y+1,height);
    int a=y0*width+x0,b=y0*width+x1,c=y1*width+x0,d=y1*width+x1;
    double first=sample(gradient,a,b,c,d,fx,fy,ix,iy);
    double second=sample(gradient+n,a,b,c,d,fx,fy,ix,iy);
    double a00=sample(hessian,a,b,c,d,fx,fy,ix,iy);
    double a11=sample(hessian+n,a,b,c,d,fx,fy,ix,iy);
    double a01=sample(hessian+2*n,a,b,c,d,fx,fy,ix,iy);
    double j11=jacobian[i],j21=jacobian[n+i],j12=jacobian[2*n+i],j22=jacobian[3*n+i];
    double determinant=j11*j22-j12*j21;
    output_gradient[i]=gradient[i]+determinant*(first*j11+second*j21);
    output_gradient[n+i]=gradient[n+i]+determinant*(first*j12+second*j22);
    double tmp1=a00*j11+a01*j21,tmp2=a01*j11+a11*j21;
    double tmp3=a00*j12+a01*j22,tmp4=a01*j12+a11*j22;
    output_hessian[i]=hessian[i]+determinant*(tmp1*j11+tmp2*j21);
    output_hessian[n+i]=hessian[n+i]+determinant*(tmp3*j12+tmp4*j22);
    output_hessian[2*n+i]=hessian[2*n+i]+determinant*(tmp1*j12+tmp2*j22);
}
"""


@lru_cache(maxsize=8)
def _kernel(device):
    "按设备缓存五通道采样核，并关闭隐式 FMA。"
    import cupy as cp

    return cp.RawModule(
        code=SOURCE, options=("--std=c++11", "--fmad=false")
    ).get_function("transport_accumulate")


def transport_accumulate(gradient, hessian, mapping, jacobian):
    "只读输入并写入独立输出，避免采样与写入竞争。"
    import cupy as cp

    height, width = gradient.shape[1:]
    out_gradient = cp.empty_like(gradient)
    out_hessian = cp.empty_like(hessian)
    _kernel(cp.cuda.Device().id)(
        ((height * width + 255) // 256,),
        (256,),
        (
            gradient,
            hessian,
            mapping,
            jacobian,
            np.int32(height),
            np.int32(width),
            out_gradient,
            out_hessian,
        ),
    )
    return out_gradient, out_hessian
