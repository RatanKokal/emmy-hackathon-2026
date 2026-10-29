#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* add_c1, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in0__u0 = add_c1[0];
        __half in1__u0 = x[a0 * 4];
        __half v0__u0 = in0__u0 * in1__u0;
        __half in0__u1 = add_c1[0];
        __half in1__u1 = x[a0 * 4 + 1];
        __half v0__u1 = in0__u1 * in1__u1;
        __half in0__u2 = add_c1[0];
        __half in1__u2 = x[a0 * 4 + 2];
        __half v0__u2 = in0__u2 * in1__u2;
        __half in0__u3 = add_c1[0];
        __half in1__u3 = x[a0 * 4 + 3];
        __half v0__u3 = in0__u3 * in1__u3;
        float4 _vs_v0__u0_d59b685d = make_float4(__half2float(v0__u0), __half2float(v0__u1), __half2float(v0__u2), __half2float(v0__u3));
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v0__u0_d59b685d;
}
