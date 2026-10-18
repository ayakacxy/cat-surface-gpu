"使用共享内存依赖闭包和 CUDA 计算图执行保序球面平滑。"

from functools import lru_cache
import hashlib
import os
from threading import RLock

import numpy as np


def enabled():
    "默认启用已接入优化，允许显式选择原 CUDA 实现做诊断。"
    value = os.environ.get("CAT_SURFACE_CUDA_OPTIMIZED", "1")
    if value not in ("0", "1"):
        raise ValueError("CAT_SURFACE_CUDA_OPTIMIZED must be 0 or 1")
    return value == "1"


_SOURCE = r"""
__device__ float3 add3(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ float3 sub3(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ float3 mul3(float3 a,float b){return make_float3(a.x*b,a.y*b,a.z*b);}
extern "C" __global__ void tiled(
 const float* points,const int* ids,const unsigned* triangles,const unsigned short* counts,
 const short* owned,const unsigned short* offsets,const unsigned short* rings,
 const signed char* selected,double strength,float* output,int W,int D,int O,int C,int R,int distance){
 int patch=blockIdx.x,lane=threadIdx.x;
 extern __shared__ float3 positions[];
 for(int i=lane;i<W;i+=blockDim.x){
  int id=ids[patch*W+i];
  positions[i]=id>=0?make_float3(points[id*3],points[id*3+1],points[id*3+2]):make_float3(0,0,0);
 }
 __syncthreads();
 for(int color=0;color<C;++color){
  int begin=offsets[patch*(C+1)+color],end=offsets[patch*(C+1)+color+1];
  for(int i=begin+lane;i<end;i+=blockDim.x){
   int row=patch*W+i;
   if(counts[row]<=1)continue;
   if(distance){
    if(!selected[ids[row]])continue;
    const double old_x = (double)positions[i].x;
    const double old_y = (double)positions[i].y;
    const double old_z = (double)positions[i].z;
    double total = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    double weighted_z = 0.0;
    for (int slot = 0; slot < counts[row]; ++slot) {
        const int neighbour = rings[row*R+slot];
        const double x = (double)positions[neighbour].x;
        const double y = (double)positions[neighbour].y;
        const double z = (double)positions[neighbour].z;
        const double distance = fabs(x - old_x) + fabs(y - old_y) + fabs(z - old_z);
        total += distance;
        if (distance > 0.0) {
            weighted_x += distance * x;
            weighted_y += distance * y;
            weighted_z += distance * z;
        }
    }
    if (total <= 0.0) continue;
    const double inverse = 1.0 - strength;
    positions[i].x = (float)(old_x * inverse + weighted_x / total * strength);
    positions[i].y = (float)(old_y * inverse + weighted_y / total * strength);
    positions[i].z = (float)(old_z * inverse + weighted_z / total * strength);
   }else{
    float total=0;float3 weighted=make_float3(0,0,0);
    for(int k=0;k<counts[row];++k){
     unsigned packed=triangles[row*D+k];
     float3 a=positions[packed&1023u],b=positions[(packed>>10)&1023u],c=positions[packed>>20];
     float3 u=sub3(b,a),v=sub3(c,a);
     float x=u.y*v.z-u.z*v.y,y=u.z*v.x-u.x*v.z,z=u.x*v.y-u.y*v.x;
     float area=0.5f*sqrtf((x*x+y*y)+z*z);
     area=isfinite(area)?fminf(fmaxf(area,0.0f),1.0f):0.0f;
     total+=area;
     weighted.x+=area*((a.x+b.x)+c.x)/3.0f;
     weighted.y+=area*((a.y+b.y)+c.y)/3.0f;
     weighted.z+=area*((a.z+b.z)+c.z)/3.0f;
    }
    if(total>0)positions[i]=make_float3(weighted.x/total,weighted.y/total,weighted.z/total);
   }
  }
  __syncthreads();
 }
 for(int o=lane;o<O;o+=blockDim.x){
  int i=owned[patch*O+o];if(i<0)continue;
  int v=ids[patch*W+i];float3 p=positions[i];
  output[v*3]=p.x;output[v*3+1]=p.y;output[v*3+2]=p.z;
 }
}
"""


@lru_cache(maxsize=8)
def _kernel(device):
    "按设备编译内核，不以快速数学改变数值语义。"
    import cupy as cp

    kernel = cp.RawKernel(_SOURCE, "tiled", options=("--std=c++11", "--fmad=false"))
    kernel.compile()
    return kernel


class TiledPlan:
    "计划仅保留固定整数表，每次调用单独分配坐标和临时缓冲。"

    def __init__(self, faces, topology):
        import cupy as cp
        from cat_surface_gpu.templates import fixed_dependency_tables

        arrays = fixed_dependency_tables(faces, topology)
        self.tables = tuple(cp.asarray(a) for a in arrays)
        ids, triangles, counts, owned, offsets, rings = arrays
        self.dimensions = tuple(
            np.int32(x)
            for x in (
                ids.shape[1],
                triangles.shape[2],
                owned.shape[1],
                offsets.shape[1] - 1,
                rings.shape[2],
            )
        )
        self.blocks = ids.shape[0]
        self.kernel = _kernel(cp.cuda.Device().id)
        self.ready = cp.cuda.get_current_stream().record()

    def smooth(
        self, points, iterations, *, selected=None, strength=1.0, distance=False
    ):
        "按批次重放，通过事件衔接调用流与捕获流，无跨线程块屏障。"
        import cupy as cp

        count = max(0, int(iterations) - 1)
        if not count:
            return
        caller = cp.cuda.get_current_stream()
        caller.wait_event(self.ready)
        selected = (
            cp.ones(len(points), cp.int8)
            if selected is None
            else cp.asarray(selected, dtype=cp.int8)
        )
        scratch = cp.empty_like(points)
        ids, triangles, counts, owned, offsets, rings = self.tables

        def step(source, destination):
            self.kernel(
                (self.blocks,),
                (128,),
                (
                    source,
                    ids,
                    triangles,
                    counts,
                    owned,
                    offsets,
                    rings,
                    selected,
                    np.float64(strength),
                    destination,
                    *self.dimensions,
                    np.int32(distance),
                ),
                shared_mem=int(self.dimensions[0]) * 12,
            )

        stream = cp.cuda.Stream(non_blocking=True)
        stream.wait_event(caller.record())
        batch = min(250, count - count % 2)
        with stream:
            if batch:
                stream.begin_capture()
                for _ in range(batch // 2):
                    step(points, scratch)
                    step(scratch, points)
                graph = stream.end_capture()
                for _ in range(count // batch):
                    graph.launch(stream)
                count %= batch
            for _ in range(count // 2):
                step(points, scratch)
                step(scratch, points)
            if count % 2:
                step(points, scratch)
                cp.copyto(points, scratch)
        # 释放捕获图和私有缓冲前等待完成，包括非默认调用流。
        stream.synchronize()


_PLANS = {}
_PLAN_LOCK = RLock()


def get_plan(faces, topology):
    "按连接关系、颜色顺序和设备限制常驻拓扑缓存数量。"
    import cupy as cp

    digest = hashlib.sha256()
    for array in (
        faces,
        topology.neighbours,
        topology.neighbour_mask,
        topology.incident_faces,
        topology.incident_mask,
        *topology.color_groups,
    ):
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (cp.cuda.Device().id, digest.digest())
    with _PLAN_LOCK:
        if key not in _PLANS:
            if len(_PLANS) >= 2:
                _PLANS.pop(next(iter(_PLANS)))
            _PLANS[key] = TiledPlan(faces, topology)
        return _PLANS[key]
