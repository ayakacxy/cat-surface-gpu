"融合初始旋转的面查询、最近点插值和逐点平方误差。"

from functools import lru_cache
import numpy as np


SOURCE = r"""
// 保留 FP32 几何及 FP64 查询，并关闭隐式 FMA。
__device__ double segment(const double* q,const float* a,const float* b,double* p) {
    float dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
    float denominator=(dx*dx+dy*dy)+dz*dz;
    double fraction=denominator!=0.0f?
        (((q[0]-a[0])*dx+(q[1]-a[1])*dy)+(q[2]-a[2])*dz)/denominator:0.0;
    fraction=fmin(1.0,fmax(0.0,fraction));
    p[0]=a[0]+fraction*dx;p[1]=a[1]+fraction*dy;p[2]=a[2]+fraction*dz;
    double x=q[0]-p[0],y=q[1]-p[1],z=q[2]-p[2];
    return (x*x+y*y)+z*z;
}
extern "C" __global__ void rotation_errors(const double* rotated,const double* source,
    const double* values,const float* triangles,const float* edge_a,const float* edge_b,
    const float* normals,const float* normal_sq,const float* xx,const float* xy,const float* yy,
    const int* offsets,const int* candidates,double scale,int grid,int packed_count,
    int total_points,int start_point,int chunk,int angles,double* errors) {
    int item=blockIdx.x*blockDim.x+threadIdx.x;
    if(item>=angles*chunk)return;
    int angle=item/chunk,point=start_point+item%chunk;
    const double* q=rotated+((long long)angle*total_points+point)*3;
    int ix=min(grid-1,max(0,(int)floor((q[0]/scale+1.0)*(grid/2.0))));
    int iy=min(grid-1,max(0,(int)floor((q[1]/scale+1.0)*(grid/2.0))));
    int iz=min(grid-1,max(0,(int)floor((q[2]/scale+1.0)*(grid/2.0))));
    int cell=ix+grid*(iy+grid*iz),begin=offsets[cell],end=offsets[cell+1];
    // 空单元保留数组实现 argmin 选择首项的行为。
    if(end==begin)end=begin+1;
    double best_distance=__longlong_as_double(0x7ff0000000000000LL),best_point[3]={};int best_face=-1;
    for(int pos=begin;pos<end;++pos) {
        int face=candidates[min(pos,packed_count-1)];
        const float* tri=triangles+face*9;
        const float* normal=normals+face*3;
        const float* ea=edge_a+face*3;const float* eb=edge_b+face*3;
        double numerator=(((tri[0]-q[0])*normal[0]+(tri[1]-q[1])*normal[1])+
                          (tri[2]-q[2])*normal[2]);
        double plane_t=normal_sq[face]>0.0f?numerator/normal_sq[face]:0.0;
        double p[3]={q[0]+plane_t*normal[0],q[1]+plane_t*normal[1],q[2]+plane_t*normal[2]};
        double ox=p[0]-tri[0],oy=p[1]-tri[1],oz=p[2]-tri[2];
        double xv=(ea[0]*ox+ea[1]*oy)+ea[2]*oz;
        double yv=(eb[0]*ox+eb[1]*oy)+eb[2]*oz;
        float denominator=xx[face]*yy[face]-xy[face]*xy[face];
        double x=denominator!=0.0f?(xv*yy[face]-yv*xy[face])/denominator:0.0;
        double y=denominator!=0.0f?(yv*xx[face]-xv*xy[face])/denominator:0.0;
        bool inside=denominator!=0.0f && x>=-1e-3 && x<=1.0+1e-3 &&
            y>=-1e-3 && y<=1.0+1e-3 && x+y>=-1e-3 && x+y<=1.0+1e-3;
        double distance;
        if(inside)distance=plane_t*plane_t*normal_sq[face];
        else {
            double vd[3];
            for(int v=0;v<3;++v) {
                double dx=q[0]-tri[v*3],dy=q[1]-tri[v*3+1],dz=q[2]-tri[v*3+2];
                vd[v]=(dx*dx+dy*dy)+dz*dz;
            }
            int closest=0;
            if(vd[1]<vd[closest])closest=1;
            if(vd[2]<vd[closest])closest=2;
            double first[3],second[3];
            double d1=segment(q,tri+((closest+2)%3)*3,tri+closest*3,first);
            double d2=segment(q,tri+closest*3,tri+((closest+1)%3)*3,second);
            distance=d1<d2?d1:d2;
            for(int a=0;a<3;++a)p[a]=d1<d2?first[a]:second[a];
        }
        if(distance<best_distance) {
            best_distance=distance;best_face=face;
            for(int a=0;a<3;++a)best_point[a]=p[a];
        }
    }
    if(best_face<0){errors[item]=__longlong_as_double(0x7ff8000000000000LL);return;}
    int f=best_face;const float* a=triangles+f*9;
    const float* ea=edge_a+f*3;const float* eb=edge_b+f*3;
    double ox=best_point[0]-a[0],oy=best_point[1]-a[1],oz=best_point[2]-a[2];
    double d20=(ox*ea[0]+oy*ea[1])+oz*ea[2];
    double d21=(ox*eb[0]+oy*eb[1])+oz*eb[2];
    float denominator=xx[f]*yy[f]-xy[f]*xy[f];
    double w1=denominator!=0.0f?(yy[f]*d20-xy[f]*d21)/denominator:0.0;
    double w2=denominator!=0.0f?(xx[f]*d21-xy[f]*d20)/denominator:0.0;
    double w0=(1.0-w1)-w2;
    double sampled=(values[f*3]*w0+values[f*3+1]*w1)+values[f*3+2]*w2;
    double delta=source[point]-sampled;errors[item]=delta*delta;
}
"""


@lru_cache(maxsize=8)
def _kernel(device):
    "按设备缓存融合内核，保留原浮点精度。"
    import cupy as cp

    return cp.RawModule(
        code=SOURCE, options=("--std=c++11", "--fmad=false")
    ).get_function("rotation_errors")


def rotation_errors(
    rotated,
    source,
    values,
    triangles,
    edge_a,
    edge_b,
    normals,
    normal_sq,
    xx,
    xy,
    yy,
    offsets,
    candidates,
    scale,
    grid,
    start,
    stop,
):
    "按既有分块返回逐点误差，沿用原 CuPy 归约。"
    import cupy as cp

    angles, total = rotated.shape[:2]
    chunk = stop - start
    errors = cp.empty((angles, chunk), dtype=cp.float64)
    _kernel(cp.cuda.Device().id)(
        ((angles * chunk + 127) // 128,),
        (128,),
        (
            rotated,
            source,
            values,
            triangles,
            edge_a,
            edge_b,
            normals,
            normal_sq,
            xx,
            xy,
            yy,
            offsets,
            candidates,
            np.float64(scale),
            np.int32(grid),
            np.int32(candidates.size),
            np.int32(total),
            np.int32(start),
            np.int32(chunk),
            np.int32(angles),
            errors,
        ),
    )
    return errors
