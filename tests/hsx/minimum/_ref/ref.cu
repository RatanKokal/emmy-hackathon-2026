extern "C" __global__
__launch_bounds__(256) void k_minimum_pointwise(const float* x, const float* zeros_like_scalar, float* minimum) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float in0__u0 = zeros_like_scalar[0];
        float v0__u0 = fminf(in0__u0, in1__u0);
        float in1__u1 = x[a0 * 4 + 1];
        float in0__u1 = zeros_like_scalar[0];
        float v0__u1 = fminf(in0__u1, in1__u1);
        float in1__u2 = x[a0 * 4 + 2];
        float in0__u2 = zeros_like_scalar[0];
        float v0__u2 = fminf(in0__u2, in1__u2);
        float in1__u3 = x[a0 * 4 + 3];
        float in0__u3 = zeros_like_scalar[0];
        float v0__u3 = fminf(in0__u3, in1__u3);
        float4 _vs_v0__u0_ed90968d = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&minimum[a0 * 4]) = _vs_v0__u0_ed90968d;
}
