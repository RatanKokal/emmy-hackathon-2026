extern "C" __global__
__launch_bounds__(256) void k_rsqrt_pointwise(const float* x, const float* add_c1, float* rsqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float v0__u0 = fabsf(in1__u0);
        float in0__u0 = add_c1[0];
        float v1__u0 = in0__u0 + v0__u0;
        float v2__u0 = rsqrtf(v1__u0);
        float in1__u1 = x[a0 * 4 + 1];
        float v0__u1 = fabsf(in1__u1);
        float in0__u1 = add_c1[0];
        float v1__u1 = in0__u1 + v0__u1;
        float v2__u1 = rsqrtf(v1__u1);
        float in1__u2 = x[a0 * 4 + 2];
        float v0__u2 = fabsf(in1__u2);
        float in0__u2 = add_c1[0];
        float v1__u2 = in0__u2 + v0__u2;
        float v2__u2 = rsqrtf(v1__u2);
        float in1__u3 = x[a0 * 4 + 3];
        float v0__u3 = fabsf(in1__u3);
        float in0__u3 = add_c1[0];
        float v1__u3 = in0__u3 + v0__u3;
        float v2__u3 = rsqrtf(v1__u3);
        float4 _vs_v2__u0_b8cce35b = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&rsqrt[a0 * 4]) = _vs_v2__u0_b8cce35b;
}
