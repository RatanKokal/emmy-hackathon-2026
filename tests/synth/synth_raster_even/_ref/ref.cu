extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, const float* y, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _rsub = _gid;
        int a0 = (_rsub / 256) * 4 + _rsub % 4;
        int a1 = (_rsub % 256) / 4;
        float in0__u0 = x[a0 * 256 + a1 * 4];
        float in1__u0 = y[a1 * 4];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = x[a0 * 256 + (a1 * 4 + 1)];
        float in1__u1 = y[a1 * 4 + 1];
        float v0__u1 = in0__u1 + in1__u1;
        float in0__u2 = x[a0 * 256 + (a1 * 4 + 2)];
        float in1__u2 = y[a1 * 4 + 2];
        float v0__u2 = in0__u2 + in1__u2;
        float in0__u3 = x[a0 * 256 + (a1 * 4 + 3)];
        float in1__u3 = y[a1 * 4 + 3];
        float v0__u3 = in0__u3 + in1__u3;
        float4 _vs_v0__u0_10e7ebcc = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&add[a0 * 256 + a1 * 4]) = _vs_v0__u0_10e7ebcc;
}
