extern "C" __global__
__launch_bounds__(256) void k_sin_pointwise(const float* x, float* sin) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v0__u0 = sinf(in0__u0);
        float v0__u1 = sinf(in0__u1);
        float v0__u2 = sinf(in0__u2);
        float v0__u3 = sinf(in0__u3);
        float4 _vs_v0__u0_ab741617 = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&sin[a0 * 4]) = _vs_v0__u0_ab741617;
}
