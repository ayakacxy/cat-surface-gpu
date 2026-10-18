"CAT 几何核共享的轻量 CUDA 向量运算。"

HEADER = r"""
struct V3 {
 float x,y,z;
 __device__ V3(float a=0):x(a),y(a),z(a){}
 __device__ V3(float a,float b,float c):x(a),y(b),z(c){}
 __device__ float& operator[](int i){return (&x)[i];}
 __device__ float operator[](int i)const{return (&x)[i];}
};
struct V4 {
 float x,y,z,w;
 __device__ V4(float a,float b,float c,float d):x(a),y(b),z(c),w(d){}
 __device__ V3 tail()const{return V3(y,z,w);}
};
struct I3 {
 int x,y,z;
 __device__ I3(int a):x(a),y(a),z(a){}
 __device__ I3(int a,int b,int c):x(a),y(b),z(c){}
 __device__ I3(V3 a):x((int)a.x),y((int)a.y),z((int)a.z){}
 __device__ int operator[](int i)const{return (&x)[i];}
};
__device__ V3 operator+(V3 a,V3 b){return V3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ V3 operator-(V3 a,V3 b){return V3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ V3 operator*(V3 a,float b){return V3(a.x*b,a.y*b,a.z*b);}
__device__ V3 operator*(float b,V3 a){return a*b;}
__device__ V3 operator/(V3 a,float b){return V3(a.x/b,a.y/b,a.z/b);}
__device__ I3 operator+(I3 a,I3 b){return I3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ I3 operator-(I3 a,I3 b){return I3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ I3 min(I3 a,I3 b){return I3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));}
__device__ I3 max(I3 a,I3 b){return I3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));}
__device__ I3 clamp(I3 a,I3 lo,I3 hi){return min(max(a,lo),hi);}
__device__ V3 floor(V3 a){return V3(floorf(a.x),floorf(a.y),floorf(a.z));}
__device__ float dot(V3 a,V3 b){return (a.x*b.x+a.y*b.y)+a.z*b.z;}
__device__ V3 cross(V3 a,V3 b){return V3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float length(V3 a){return sqrtf(dot(a,a));}
"""
