"用于 Metal CAT 初始旋转的融合 FP32 最近面代价。"

from functools import lru_cache
import numpy as np
from .surf2sphere import _mlx

HEADER = r"""inline float segment(thread const float* q,device const float* a,device const float* b,thread float* p) {
    float dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
    float denominator=(dx*dx+dy*dy)+dz*dz;
    float fraction=denominator!=0.0f?
        (((q[0]-a[0])*dx+(q[1]-a[1])*dy)+(q[2]-a[2])*dz)/denominator:0.0;
    fraction=metal::min(1.0,metal::max(0.0,fraction));
    p[0]=a[0]+fraction*dx;p[1]=a[1]+fraction*dy;p[2]=a[2]+fraction*dz;
    float x=q[0]-p[0],y=q[1]-p[1],z=q[2]-p[2];
    return (x*x+y*y)+z*z;
}
"""

SOURCE = r"""
int item=int(thread_position_in_grid.x),total_points=int(points_shape[0]),angles=int(matrices_shape[0]);
if(item>=angles*total_points)return;
int angle=item/total_points,point=item%total_points;
int grid=settings[0],packed_count=int(candidates_shape[0]);float scale=scale_value[0];
float p0=points[point*3],p1=points[point*3+1],p2=points[point*3+2];
float q[3];for(int axis=0;axis<3;++axis)q[axis]=(p0*matrices[angle*9+axis*3]+p1*matrices[angle*9+axis*3+1])+p2*matrices[angle*9+axis*3+2];
    int ix=metal::min(grid-1,metal::max(0,(int)metal::floor((q[0]/scale+1.0)*(grid/2.0))));
    int iy=metal::min(grid-1,metal::max(0,(int)metal::floor((q[1]/scale+1.0)*(grid/2.0))));
    int iz=metal::min(grid-1,metal::max(0,(int)metal::floor((q[2]/scale+1.0)*(grid/2.0))));
    int cell=ix+grid*(iy+grid*iz),begin=offsets[cell],end=offsets[cell+1];
    // 空单元保留数组实现 argmin 选择首项的行为。
    if(end==begin)end=begin+1;
    float best_distance=INFINITY,best_point[3]={};int best_face=-1;
    for(int pos=begin;pos<end;++pos) {
        int face=candidates[metal::min(pos,packed_count-1)];
        device const float* tri=triangles+face*9;
        device const float* normal=normals+face*3;
        device const float* ea=edge_a+face*3;device const float* eb=edge_b+face*3;
        float numerator=(((tri[0]-q[0])*normal[0]+(tri[1]-q[1])*normal[1])+
                          (tri[2]-q[2])*normal[2]);
        float plane_t=normal_sq[face]>0.0f?numerator/normal_sq[face]:0.0;
        float p[3]={q[0]+plane_t*normal[0],q[1]+plane_t*normal[1],q[2]+plane_t*normal[2]};
        float ox=p[0]-tri[0],oy=p[1]-tri[1],oz=p[2]-tri[2];
        float xv=(ea[0]*ox+ea[1]*oy)+ea[2]*oz;
        float yv=(eb[0]*ox+eb[1]*oy)+eb[2]*oz;
        float denominator=xx[face]*yy[face]-xy[face]*xy[face];
        float x=denominator!=0.0f?(xv*yy[face]-yv*xy[face])/denominator:0.0;
        float y=denominator!=0.0f?(yv*xx[face]-xv*xy[face])/denominator:0.0;
        bool inside=denominator!=0.0f && x>=-1e-3 && x<=1.0+1e-3 &&
            y>=-1e-3 && y<=1.0+1e-3 && x+y>=-1e-3 && x+y<=1.0+1e-3;
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
    if(best_face<0){errors[item]=NAN;return;}
    int f=best_face;device const float* a=triangles+f*9;
    device const float* ea=edge_a+f*3;device const float* eb=edge_b+f*3;
    float ox=best_point[0]-a[0],oy=best_point[1]-a[1],oz=best_point[2]-a[2];
    float d20=(ox*ea[0]+oy*ea[1])+oz*ea[2];
    float d21=(ox*eb[0]+oy*eb[1])+oz*eb[2];
    float denominator=xx[f]*yy[f]-xy[f]*xy[f];
    float w1=denominator!=0.0f?(yy[f]*d20-xy[f]*d21)/denominator:0.0;
    float w2=denominator!=0.0f?(xx[f]*d21-xy[f]*d20)/denominator:0.0;
    float w0=(1.0-w1)-w2;
    float sampled=(values[f*3]*w0+values[f*3+1]*w1)+values[f*3+2]*w2;
    float delta=source[point]-sampled;errors[item]=delta*delta;
"""


@lru_cache(maxsize=1)
def _kernel():
    mx = _mlx()
    return mx.fast.metal_kernel(
        name="cat_rotation_cost_fp32",
        input_names=[
            "points",
            "matrices",
            "source",
            "values",
            "triangles",
            "edge_a",
            "edge_b",
            "normals",
            "normal_sq",
            "xx",
            "xy",
            "yy",
            "offsets",
            "candidates",
            "scale_value",
            "settings",
        ],
        output_names=["errors"],
        header=HEADER,
        source=SOURCE,
        compile_options={"math_mode": "safe"},
    )


class RotationCosts:
    "目标几何常驻设备，每个角度向 CPU 返回一个代价。"

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
        mx = _mlx()
        with mx.stream(mx.gpu):
            self.points = mx.array(np.asarray(points, dtype=np.float32))
            self.arrays = [
                mx.array(np.ascontiguousarray(a, dtype=np.float32))
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
            ]
            self.arrays.extend(
                [
                    mx.array(np.asarray(offsets, dtype=np.int32)),
                    mx.array(np.asarray(candidates, dtype=np.int32)),
                    mx.array([scale], mx.float32),
                    mx.array([grid], mx.int32),
                ]
            )
            mx.eval(self.points, self.arrays)

    def __call__(self, matrices):
        mx = _mlx()
        matrix = np.ascontiguousarray(matrices, dtype=np.float32).reshape(-1, 3, 3)
        with mx.stream(mx.gpu):
            shape = (len(matrix), len(self.points))
            errors = _kernel()(
                inputs=[self.points, mx.array(matrix), *self.arrays],
                output_shapes=[shape],
                output_dtypes=[mx.float32],
                grid=(shape[0] * shape[1], 1, 1),
                threadgroup=(128, 1, 1),
                stream=mx.gpu,
            )[0]
            result = mx.sum(errors, axis=1)
            mx.eval(result)
            output = np.array(result).astype(np.float64)
        if not np.isfinite(output).all():
            raise FloatingPointError("CAT Metal rotation produced non-finite costs")
        return output
