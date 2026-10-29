extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* add_c1, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = add_c1[0];
        float in1__u0 = x[a0 * 4];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = add_c1[0];
        float in1__u1 = x[a0 * 4 + 1];
        float v0__u1 = in0__u1 + in1__u1;
        float in0__u2 = add_c1[0];
        float in1__u2 = x[a0 * 4 + 2];
        float v0__u2 = in0__u2 + in1__u2;
        float in0__u3 = add_c1[0];
        float in1__u3 = x[a0 * 4 + 3];
        float v0__u3 = in0__u3 + in1__u3;
        float negv__u0 = 0.0f - in1__u0;
        if (in1__u0 > 0.0f) {
            add[a0 * 4] = in1__u0;
        } else {
            add[a0 * 4] = negv__u0;
        }
        float negv__u1 = 0.0f - in1__u1;
        if (in1__u1 > 0.0f) {
            add[a0 * 4 + 1] = in1__u1;
        } else {
            add[a0 * 4 + 1] = negv__u1;
        }
        float negv__u2 = 0.0f - in1__u2;
        if (in1__u2 > 0.0f) {
            add[a0 * 4 + 2] = in1__u2;
        } else {
            add[a0 * 4 + 2] = negv__u2;
        }
        float negv__u3 = 0.0f - in1__u3;
        if (in1__u3 > 0.0f) {
            add[a0 * 4 + 3] = in1__u3;
        } else {
            add[a0 * 4 + 3] = negv__u3;
        }
}
