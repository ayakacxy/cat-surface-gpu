"融合 FP32 旋转与最近面代价，同时保留 CPU 搜索控制。"

from functools import lru_cache
import numpy as np

_SOURCE = r"""
__device__ inline float segment(const float* q,const float* a,const float* b,float* p) {
    float dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
    float denominator=(dx*dx+dy*dy)+dz*dz;
    float fraction=denominator!=0.0f?
        (((q[0]-a[0])*dx+(q[1]-a[1])*dy)+(q[2]-a[2])*dz)/denominator:0.0f;
    fraction=min(1.0f,max(0.0f,fraction));
    p[0]=a[0]+fraction*dx;p[1]=a[1]+fraction*dy;p[2]=a[2]+fraction*dz;
    float x=q[0]-p[0],y=q[1]-p[1],z=q[2]-p[2];
    return (x*x+y*y)+z*z;
}

extern "C" __global__ void costs(const float* points,const float* matrices,const float* source,const float* values,const float* triangles,const float* edge_a,const float* edge_b,const float* normals,const float* normal_sq,const float* xx,const float* xy,const float* yy,const int* offsets,const int* candidates,float scale,int grid,int packed_count,int total_points,int angles,float* errors){
int item=blockIdx.x*blockDim.x+threadIdx.x;
if(item>=angles*total_points)return;
int angle=item/total_points,point=item%total_points;

float p0=points[point*3],p1=points[point*3+1],p2=points[point*3+2];
float q[3];for(int axis=0;axis<3;++axis)q[axis]=(p0*matrices[angle*9+axis*3]+p1*matrices[angle*9+axis*3+1])+p2*matrices[angle*9+axis*3+2];
    int ix=min(grid-1,max(0,(int)floor((q[0]/scale+1.0f)*(grid/2.0f))));
    int iy=min(grid-1,max(0,(int)floor((q[1]/scale+1.0f)*(grid/2.0f))));
    int iz=min(grid-1,max(0,(int)floor((q[2]/scale+1.0f)*(grid/2.0f))));
    int cell=ix+grid*(iy+grid*iz),begin=offsets[cell],end=offsets[cell+1];
    // 空单元保留原始候选选择规则。
    if(end==begin)end=begin+1;
    float best_distance=__int_as_float(0x7f800000),best_point[3]={};int best_face=-1;
    for(int pos=begin;pos<end;++pos) {
        int face=candidates[min(pos,packed_count-1)];
        const float* tri=triangles+face*9;
        const float* normal=normals+face*3;
        const float* ea=edge_a+face*3;const float* eb=edge_b+face*3;
        float numerator=(((tri[0]-q[0])*normal[0]+(tri[1]-q[1])*normal[1])+
                          (tri[2]-q[2])*normal[2]);
        float plane_t=normal_sq[face]>0.0f?numerator/normal_sq[face]:0.0f;
        float p[3]={q[0]+plane_t*normal[0],q[1]+plane_t*normal[1],q[2]+plane_t*normal[2]};
        float ox=p[0]-tri[0],oy=p[1]-tri[1],oz=p[2]-tri[2];
        float xv=(ea[0]*ox+ea[1]*oy)+ea[2]*oz;
        float yv=(eb[0]*ox+eb[1]*oy)+eb[2]*oz;
        float denominator=xx[face]*yy[face]-xy[face]*xy[face];
        float x=denominator!=0.0f?(xv*yy[face]-yv*xy[face])/denominator:0.0f;
        float y=denominator!=0.0f?(yv*xx[face]-xv*xy[face])/denominator:0.0f;
        bool inside=denominator!=0.0f && x>=-1e-3f && x<=1.0f+1e-3f &&
            y>=-1e-3f && y<=1.0f+1e-3f && x+y>=-1e-3f && x+y<=1.0f+1e-3f;
        float distance;
        if(inside)distance=plane_t*plane_t*normal_sq[face];
        else {
            float vd[3];
            for(int v=0;v<3;++v) {
                float dx=q[0]-tri[v*3],dy=q[1]-tri[v*3+1],dz=q[2]-tri[v*3+2];
                vd[v]=(dx*dx+dy*dy)+dz*dz;
            }
            int closest=0;
            if(vd[1]<vd[closest])closest=1;
            if(vd[2]<vd[closest])closest=2;
            float first[3],second[3];
            float d1=segment(q,tri+((closest+2)%3)*3,tri+closest*3,first);
            float d2=segment(q,tri+closest*3,tri+((closest+1)%3)*3,second);
            distance=d1<d2?d1:d2;
            for(int a=0;a<3;++a)p[a]=d1<d2?first[a]:second[a];
        }
        if(distance<best_distance) {
            best_distance=distance;best_face=face;
            for(int a=0;a<3;++a)best_point[a]=p[a];
        }
    }
    if(best_face<0){errors[item]=__int_as_float(0x7fffffff);return;}
    int f=best_face;const float* a=triangles+f*9;
    const float* ea=edge_a+f*3;const float* eb=edge_b+f*3;
    float ox=best_point[0]-a[0],oy=best_point[1]-a[1],oz=best_point[2]-a[2];
    float d20=(ox*ea[0]+oy*ea[1])+oz*ea[2];
    float d21=(ox*eb[0]+oy*eb[1])+oz*eb[2];
    float denominator=xx[f]*yy[f]-xy[f]*xy[f];
    float w1=denominator!=0.0f?(yy[f]*d20-xy[f]*d21)/denominator:0.0f;
    float w2=denominator!=0.0f?(xx[f]*d21-xy[f]*d20)/denominator:0.0f;
    float w0=(1.0f-w1)-w2;
    float sampled=(values[f*3]*w0+values[f*3+1]*w1)+values[f*3+2]*w2;
    float delta=source[point]-sampled;errors[item]=delta*delta;

}
"""


@lru_cache(maxsize=8)
def _kernel(device):
    "按设备缓存融合旋转代价核。"
    import cupy as cp

    return cp.RawKernel(_SOURCE, "costs", options=("--std=c++11", "--fmad=false"))


class RotationCosts:
    "搜索期间将固定点、特征、面和索引保留在 GPU。"

    def __init__(
        self,
        points,
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
    ):
        import cupy as cp

        self.points = cp.asarray(points, dtype=cp.float32)
        self.arrays = tuple(
            cp.asarray(a, dtype=cp.float32)
            for a in (
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
            )
        )
        self.indices = (
            cp.asarray(offsets, dtype=cp.int32),
            cp.asarray(candidates, dtype=cp.int32),
        )
        self.scale, self.grid = np.float32(scale), np.int32(grid)

    def __call__(self, matrices):
        import cupy as cp

        matrix = cp.asarray(
            np.ascontiguousarray(matrices, dtype=np.float32).reshape(-1, 3, 3)
        )
        errors = cp.empty((len(matrix), len(self.points)), cp.float32)
        _kernel(cp.cuda.Device().id)(
            ((errors.size + 127) // 128,),
            (128,),
            (
                self.points,
                matrix,
                *self.arrays,
                *self.indices,
                self.scale,
                self.grid,
                np.int32(self.indices[1].size),
                np.int32(len(self.points)),
                np.int32(len(matrix)),
                errors,
            ),
        )
        result = cp.asnumpy(cp.sum(errors, axis=1)).astype(np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("CAT CUDA rotation produced non-finite costs")
        return result
