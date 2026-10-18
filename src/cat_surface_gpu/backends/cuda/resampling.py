"使用空间桶和四个最近邻建立 CUDA 球面采样模板。"

from functools import lru_cache
import numpy as np
from numba import njit
from cat_surface_gpu.backends.cuda.geometry import HEADER

_SOURCE = r"""

__device__ inline V3 load_point(const float* p,int i){return V3(p[3*i],p[3*i+1],p[3*i+2]);}
__device__ inline V4 closest(V3 p,V3 a,V3 b,V3 c){
 V3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0 && d2<=0)return V4(dot(ap,ap),1,0,0);
 V3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
 if(d3>=0 && d4<=d3)return V4(dot(bp,bp),0,1,0);
 float vc=d1*d4-d3*d2;
 if(vc<=0 && d1>=0 && d3<=0){float v=d1/(d1-d3);V3 d=p-(a+v*ab);return V4(dot(d,d),1-v,v,0);}
 V3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
 if(d6>=0 && d5<=d6)return V4(dot(cp,cp),0,0,1);
 float vb=d5*d2-d1*d6;
 if(vb<=0 && d2>=0 && d6<=0){float w=d2/(d2-d6);V3 d=p-(a+w*ac);return V4(dot(d,d),1-w,0,w);}
 float va=d3*d6-d5*d4;
 if(va<=0 && d4-d3>=0 && d5-d6>=0){float w=(d4-d3)/((d4-d3)+(d5-d6));V3 d=p-(b+w*(c-b));return V4(dot(d,d),0,1-w,w);}
 float inverse=1.0f/(va+vb+vc),v=vb*inverse,w=vc*inverse;V3 d=p-(a+ab*v+ac*w);
 return V4(dot(d,d),1-v-w,v,w);
}

extern "C" __global__ void nearest_points(const float* source,const float* target,const int* offsets,const int* ordered,const float* parameters,int* nearest,int n){

int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
V3 q=V3(target[3*i],target[3*i+1],target[3*i+2]);
V3 origin=V3(parameters[0],parameters[1],parameters[2]);float cell=parameters[3];int g=64;
I3 center=clamp(I3(floor((q-origin)/cell)),I3(0),I3(g-1));
float best[4]={__int_as_float(0x7f800000),__int_as_float(0x7f800000),__int_as_float(0x7f800000),__int_as_float(0x7f800000)};int ids[4]={-1,-1,-1,-1};
for(int radius=0;radius<g;++radius){
 I3 lo=max(center-I3(radius),I3(0)),hi=min(center+I3(radius),I3(g-1));
 for(int z=lo.z;z<=hi.z;++z)for(int y=lo.y;y<=hi.y;++y)for(int x=lo.x;x<=hi.x;++x){
  if(radius>0 && max(max(abs(x-center.x),abs(y-center.y)),abs(z-center.z))!=radius)continue;
  int bin=x+g*(y+g*z);
  for(int j=offsets[bin];j<offsets[bin+1];++j){
   int id=ordered[j];V3 p=V3(source[3*id],source[3*id+1],source[3*id+2]);
   V3 d=q-p;float distance=(d.x*d.x+d.y*d.y)+d.z*d.z;
   int slot=3;if(distance>best[slot] || (distance==best[slot] && id>=ids[slot]))continue;
   while(slot>0 && (distance<best[slot-1] || (distance==best[slot-1] && id<ids[slot-1]))){best[slot]=best[slot-1];ids[slot]=ids[slot-1];--slot;}
   best[slot]=distance;ids[slot]=id;
  }
 }
 float bound=__int_as_float(0x7f800000);
 for(int axis=0;axis<3;++axis){
  if(lo[axis]>0)bound=min(bound,q[axis]-(origin[axis]+float(lo[axis])*cell));
  if(hi[axis]<g-1)bound=min(bound,origin[axis]+float(hi[axis]+1)*cell-q[axis]);
 }
 // 边界距离采用保守余量，避免 FP32 舍入导致提前终止。
 bound=max(0.0f,bound-2e-6f*max(1.0f,cell*64));
 if(ids[3]>=0 && bound*bound>best[3])break;
}
for(int k=0;k<4;++k)nearest[i*4+k]=ids[k];

}
extern "C" __global__ void stencil_points(const float* source,const int* faces,const float* target,const int* nearest,const int* incident,const int* degree,int* selected,float* barycentric,int n,int slots){

int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
V3 q=load_point(target,i);float best=__int_as_float(0x7f800000);int face=-1;V3 weights=V3(0);
for(int k=0;k<4;++k){
 int v=nearest[i*4+k];if(v<0)continue;
 for(int j=0;j<degree[v];++j){
  int f=incident[v*slots+j];
  V4 result=closest(q,load_point(source,faces[3*f]),load_point(source,faces[3*f+1]),load_point(source,faces[3*f+2]));
  if(result.x<best){best=result.x;face=f;weights=result.tail();}
 }
}
selected[i]=face;for(int k=0;k<3;++k)barycentric[3*i+k]=weights[k];

}
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


@lru_cache(maxsize=8)
def _kernels(device):
    "按设备编译最近点和三角形投影核。"
    import cupy as cp

    module = cp.RawModule(
        code=HEADER + _SOURCE, options=("--std=c++11", "--fmad=false")
    )
    return module.get_function("nearest_points"), module.get_function("stencil_points")


def build_stencil(source, faces, target, topology=None):
    "索引当前坐标，仅复用固定整数拓扑。"
    import cupy as cp
    from cat_surface_gpu.resampling import ResamplingStencil, build_resampling_topology

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
        raise ValueError("CUDA resampling requires source (N>=4,3) and target (M,3)")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not faces.size
        or faces.min() < 0
        or faces.max() >= len(source)
    ):
        raise ValueError("CUDA resampling requires valid triangle indices")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("CUDA resampling coordinates must be finite")
    if not len(target):
        return ResamplingStencil(np.empty(0, np.int32), np.empty((0, 3), np.float64))
    origin = source.min(axis=0)
    span = float(np.max(source.max(axis=0) - origin))
    cell = np.float32(max(span, 1e-6) / 64)
    bins = np.clip(np.floor((source - origin) / cell), 0, 63).astype(np.int32)
    offsets, ordered = _pack(bins[:, 0] + 64 * (bins[:, 1] + 64 * bins[:, 2]))
    tables = (
        topology
        if topology is not None
        else build_resampling_topology(faces, len(source), binary_cache=True)
    )
    source_gpu, target_gpu = cp.asarray(source), cp.asarray(target)
    nearest = cp.empty((len(target), 4), cp.int32)
    indices = cp.empty(len(target), cp.int32)
    weights = cp.empty((len(target), 3), cp.float32)
    first, second = _kernels(cp.cuda.Device().id)
    blocks = ((len(target) + 127) // 128,)
    first(
        blocks,
        (128,),
        (
            source_gpu,
            target_gpu,
            cp.asarray(offsets),
            cp.asarray(ordered),
            cp.asarray(np.r_[origin, cell].astype(np.float32)),
            nearest,
            np.int32(len(target)),
        ),
    )
    second(
        blocks,
        (128,),
        (
            source_gpu,
            cp.asarray(faces),
            target_gpu,
            nearest,
            cp.asarray(tables.incident_faces),
            cp.asarray(tables.incident_counts),
            indices,
            weights,
            np.int32(len(target)),
            np.int32(tables.incident_faces.shape[1]),
        ),
    )
    indices, weights = cp.asnumpy(indices), cp.asnumpy(weights).astype(np.float64)
    if np.any(indices < 0) or not np.isfinite(weights).all():
        raise FloatingPointError(
            "CUDA resampling could not find finite triangle weights"
        )
    return ResamplingStencil(indices, weights)
