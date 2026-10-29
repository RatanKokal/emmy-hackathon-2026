extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, const float* zeros_like_scalar, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float v1__u0 = fmaxf(0.0f, in1__u0);
        float in0__u0 = zeros_like_scalar[0];
        float v0__u0 = fmaxf(in0__u0, in1__u0);
        float v2__u0 = v0__u0 + v1__u0;
        float in1__u1 = x[a0 * 4 + 1];
        float v1__u1 = fmaxf(0.0f, in1__u1);
        float in0__u1 = zeros_like_scalar[0];
        float v0__u1 = fmaxf(in0__u1, in1__u1);
        float v2__u1 = v0__u1 + v1__u1;
        float in1__u2 = x[a0 * 4 + 2];
        float v1__u2 = fmaxf(0.0f, in1__u2);
        float in0__u2 = zeros_like_scalar[0];
        float v0__u2 = fmaxf(in0__u2, in1__u2);
        float v2__u2 = v0__u2 + v1__u2;
        float in1__u3 = x[a0 * 4 + 3];
        float v1__u3 = fmaxf(0.0f, in1__u3);
        float in0__u3 = zeros_like_scalar[0];
        float v0__u3 = fmaxf(in0__u3, in1__u3);
        float v2__u3 = v0__u3 + v1__u3;
        float4 _vs_v2__u0_0e6b8326 = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v2__u0_0e6b8326;
}
