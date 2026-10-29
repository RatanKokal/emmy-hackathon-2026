extern "C" __global__
__launch_bounds__(256) void k_sqrt_pointwise(const float* x, float* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v0__u0 = fabsf(in0__u0);
        float v1__u0 = sqrtf(v0__u0);
        float v0__u1 = fabsf(in0__u1);
        float v1__u1 = sqrtf(v0__u1);
        float v0__u2 = fabsf(in0__u2);
        float v1__u2 = sqrtf(v0__u2);
        float v0__u3 = fabsf(in0__u3);
        float v1__u3 = sqrtf(v0__u3);
        float4 _vs_v1__u0_fb46fefc = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&sqrt[a0 * 4]) = _vs_v1__u0_fb46fefc;
}
