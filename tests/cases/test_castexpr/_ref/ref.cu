extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* add_c1, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = add_c1[a0 * 4];
        float in1__u0 = x[((i32)(in0__u0))];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = add_c1[a0 * 4 + 1];
        float in1__u1 = x[((i32)(in0__u1))];
        float v0__u1 = in0__u1 + in1__u1;
        float in0__u2 = add_c1[a0 * 4 + 2];
        float in1__u2 = x[((i32)(in0__u2))];
        float v0__u2 = in0__u2 + in1__u2;
        float in0__u3 = add_c1[a0 * 4 + 3];
        float in1__u3 = x[((i32)(in0__u3))];
        float v0__u3 = in0__u3 + in1__u3;
        float4 _vs_v0__u0_4d43c87b = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v0__u0_4d43c87b;
}
