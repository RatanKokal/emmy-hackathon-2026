extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* gt_c1, const float* x, float* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        float in1__u0 = x[a0 * 4];
        bool v0__u0 = in1__u0 > in0__u0;
        float v1__u0 = 0.0f - in1__u0;
        float v2__u0 = ((v0__u0) ? (in1__u0) : (v1__u0));
        float in0__u1 = gt_c1[0];
        float in1__u1 = x[a0 * 4 + 1];
        bool v0__u1 = in1__u1 > in0__u1;
        float v1__u1 = 0.0f - in1__u1;
        float v2__u1 = ((v0__u1) ? (in1__u1) : (v1__u1));
        float in0__u2 = gt_c1[0];
        float in1__u2 = x[a0 * 4 + 2];
        bool v0__u2 = in1__u2 > in0__u2;
        float v1__u2 = 0.0f - in1__u2;
        float v2__u2 = ((v0__u2) ? (in1__u2) : (v1__u2));
        float in0__u3 = gt_c1[0];
        float in1__u3 = x[a0 * 4 + 3];
        bool v0__u3 = in1__u3 > in0__u3;
        float v1__u3 = 0.0f - in1__u3;
        float v2__u3 = ((v0__u3) ? (in1__u3) : (v1__u3));
        float4 _vs_v2__u0_740a266d = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&where[a0 * 4]) = _vs_v2__u0_740a266d;
}
