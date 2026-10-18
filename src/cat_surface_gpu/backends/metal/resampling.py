"Metal 上的有界网格最近顶点查询和重心插值模板。"

from functools import lru_cache
import numpy as np
from numba import njit
from .surf2sphere import _mlx

_NEAREST = r"""
int i=int(thread_position_in_grid.x),n=int(target_shape[0]);if(i>=n)return;
float3 q=float3(target[3*i],target[3*i+1],target[3*i+2]);
float3 origin=float3(parameters[0],parameters[1],parameters[2]);float cell=parameters[3];int g=64;
int3 center=clamp(int3(floor((q-origin)/cell)),int3(0),int3(g-1));
float best[4]={INFINITY,INFINITY,INFINITY,INFINITY};int ids[4]={-1,-1,-1,-1};
for(int radius=0;radius<g;++radius){
 int3 lo=max(center-int3(radius),int3(0)),hi=min(center+int3(radius),int3(g-1));
 for(int z=lo.z;z<=hi.z;++z)for(int y=lo.y;y<=hi.y;++y)for(int x=lo.x;x<=hi.x;++x){
  if(radius>0 && max(max(abs(x-center.x),abs(y-center.y)),abs(z-center.z))!=radius)continue;
  int bin=x+g*(y+g*z);
  for(int j=offsets[bin];j<offsets[bin+1];++j){
   int id=ordered[j];float3 p=float3(source[3*id],source[3*id+1],source[3*id+2]);
   float3 d=q-p;float distance=(d.x*d.x+d.y*d.y)+d.z*d.z;
   int slot=3;if(distance>best[slot] || (distance==best[slot] && id>=ids[slot]))continue;
   while(slot>0 && (distance<best[slot-1] || (distance==best[slot-1] && id<ids[slot-1]))){best[slot]=best[slot-1];ids[slot]=ids[slot-1];--slot;}
   best[slot]=distance;ids[slot]=id;
  }
 }
 float bound=INFINITY;
 for(int axis=0;axis<3;++axis){
  if(lo[axis]>0)bound=min(bound,q[axis]-(origin[axis]+float(lo[axis])*cell));
  if(hi[axis]<g-1)bound=min(bound,origin[axis]+float(hi[axis]+1)*cell-q[axis]);
 }
 // 桶边界保留保守 FP32 余量。
 bound=max(0.0f,bound-2e-6f*max(1.0f,cell*64));
 if(ids[3]>=0 && bound*bound>best[3])break;
}
for(int k=0;k<4;++k)nearest[i*4+k]=ids[k];
"""
_CLOSEST = r"""
inline float3 load_point(device const float* p,int i){return float3(p[3*i],p[3*i+1],p[3*i+2]);}
inline float4 closest(float3 p,float3 a,float3 b,float3 c){
 float3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0 && d2<=0)return float4(dot(ap,ap),1,0,0);
 float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
 if(d3>=0 && d4<=d3)return float4(dot(bp,bp),0,1,0);
 float vc=d1*d4-d3*d2;
 if(vc<=0 && d1>=0 && d3<=0){float v=d1/(d1-d3);float3 d=p-(a+v*ab);return float4(dot(d,d),1-v,v,0);}
 float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
 if(d6>=0 && d5<=d6)return float4(dot(cp,cp),0,0,1);
 float vb=d5*d2-d1*d6;
 if(vb<=0 && d2>=0 && d6<=0){float w=d2/(d2-d6);float3 d=p-(a+w*ac);return float4(dot(d,d),1-w,0,w);}
 float va=d3*d6-d5*d4;
 if(va<=0 && d4-d3>=0 && d5-d6>=0){float w=(d4-d3)/((d4-d3)+(d5-d6));float3 d=p-(b+w*(c-b));return float4(dot(d,d),0,1-w,w);}
 float inverse=1.0f/(va+vb+vc),v=vb*inverse,w=vc*inverse;float3 d=p-(a+ab*v+ac*w);
 return float4(dot(d,d),1-v-w,v,w);
}
"""
_STENCIL = r"""
int i=int(thread_position_in_grid.x),n=int(target_shape[0]);if(i>=n)return;
int slots=int(incident_shape[1]);float3 q=load_point(target,i);float best=INFINITY;int face=-1;float3 weights=float3(0);
for(int k=0;k<4;++k){
 int v=nearest[i*4+k];if(v<0)continue;
 for(int j=0;j<degree[v];++j){
  int f=incident[v*slots+j];
  float4 result=closest(q,load_point(source,faces[3*f]),load_point(source,faces[3*f+1]),load_point(source,faces[3*f+2]));
  if(result.x<best){best=result.x;face=f;weights=result.yzw;}
 }
}
selected[i]=face;for(int k=0;k<3;++k)barycentric[3*i+k]=weights[k];
"""


@njit(cache=True)
def _pack(cells):
    counts = np.zeros(64**3 + 1, np.int32)
    for cell in cells:
        counts[cell + 1] += 1
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]
    cursor = counts[:-1].copy()
    ordered = np.empty(len(cells), np.int32)
    for i in range(len(cells)):
        cell = cells[i]
        ordered[cursor[cell]] = i
        cursor[cell] += 1
    return counts, ordered


@lru_cache(maxsize=1)
def _kernels():
    mx = _mlx()
    return (
        mx.fast.metal_kernel(
            name="cat_grid_nearest_fp32",
            input_names=["source", "target", "offsets", "ordered", "parameters"],
            output_names=["nearest"],
            source=_NEAREST,
            compile_options={"math_mode": "safe"},
        ),
        mx.fast.metal_kernel(
            name="cat_barycentric_fp32",
            input_names=["source", "faces", "target", "nearest", "incident", "degree"],
            output_names=["selected", "barycentric"],
            header=_CLOSEST,
            source=_STENCIL,
            compile_options={"math_mode": "safe"},
        ),
    )


def build_stencil(source, faces, target, topology=None):
    from ...resampling import ResamplingStencil, build_resampling_topology

    mx = _mlx()
    source = np.ascontiguousarray(source, np.float32)
    target = np.ascontiguousarray(target, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 4
        or target.ndim != 2
        or target.shape[1] != 3
    ):
        raise ValueError("Metal resampling requires source (N>=4,3) and target (M,3)")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or (faces.size and (faces.min() < 0 or faces.max() >= len(source)))
    ):
        raise ValueError("Metal resampling requires valid triangle indices")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Metal resampling coordinates must be finite")
    if not len(target):
        return ResamplingStencil(np.empty(0, np.int32), np.empty((0, 3), np.float64))
    origin = source.min(axis=0)
    span = float(np.max(source.max(axis=0) - origin))
    cell = np.float32(max(span, 1e-6) / 64)
    bins = np.clip(np.floor((source - origin) / cell), 0, 63).astype(np.int32)
    cells = bins[:, 0] + 64 * (bins[:, 1] + 64 * bins[:, 2])
    offsets, ordered = _pack(cells)
    tables = (
        topology
        if topology is not None
        else build_resampling_topology(faces, len(source), binary_cache=True)
    )
    with mx.stream(mx.gpu):
        source_gpu, target_gpu = mx.array(source), mx.array(target)
        nearest_kernel, stencil_kernel = _kernels()
        nearest = nearest_kernel(
            inputs=[
                source_gpu,
                target_gpu,
                mx.array(offsets),
                mx.array(ordered),
                mx.array(np.r_[origin, cell].astype(np.float32)),
            ],
            output_shapes=[(len(target), 4)],
            output_dtypes=[mx.int32],
            grid=(len(target), 1, 1),
            threadgroup=(128, 1, 1),
            stream=mx.gpu,
        )[0]
        indices, weights = stencil_kernel(
            inputs=[
                source_gpu,
                mx.array(faces),
                target_gpu,
                nearest,
                mx.array(tables.incident_faces),
                mx.array(tables.incident_counts),
            ],
            output_shapes=[(len(target),), (len(target), 3)],
            output_dtypes=[mx.int32, mx.float32],
            grid=(len(target), 1, 1),
            threadgroup=(128, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(indices, weights)
        indices = np.array(indices)
        weights = np.array(weights).astype(np.float64)
    if np.any(indices < 0) or not np.isfinite(weights).all():
        raise FloatingPointError(
            "Metal resampling could not find finite triangle weights"
        )
    return ResamplingStencil(indices, weights)
