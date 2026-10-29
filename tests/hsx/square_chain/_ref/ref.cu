extern "C" __global__
__launch_bounds__(256) void k_add_1_pointwise(const float* mul_c1, const float* x, const float* add_1_c1, float* add_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 4];
        float v0__u0 = in1__u0 * in2__u0;
        float v1__u0 = in2__u0 * in2__u0;
        float v2__u0 = v0__u0 + v1__u0;
        float in0__u0 = add_1_c1[0];
        float v3__u0 = in0__u0 + v2__u0;
        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 4 + 1];
        float v0__u1 = in1__u1 * in2__u1;
        float v1__u1 = in2__u1 * in2__u1;
        float v2__u1 = v0__u1 + v1__u1;
        float in0__u1 = add_1_c1[0];
        float v3__u1 = in0__u1 + v2__u1;
        float in1__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 4 + 2];
        float v0__u2 = in1__u2 * in2__u2;
        float v1__u2 = in2__u2 * in2__u2;
        float v2__u2 = v0__u2 + v1__u2;
        float in0__u2 = add_1_c1[0];
        float v3__u2 = in0__u2 + v2__u2;
        float in1__u3 = mul_c1[0];
        float in2__u3 = x[a0 * 4 + 3];
        float v0__u3 = in1__u3 * in2__u3;
        float v1__u3 = in2__u3 * in2__u3;
        float v2__u3 = v0__u3 + v1__u3;
        float in0__u3 = add_1_c1[0];
        float v3__u3 = in0__u3 + v2__u3;
        float4 _vs_v3__u0_35785b33 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&add_1[a0 * 4]) = _vs_v3__u0_35785b33;
}
