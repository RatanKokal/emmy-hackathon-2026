extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const float* x, float* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v0__u0 = 1.0f / (1.0f + expf(0.0f - in0__u0));
        float v0__u1 = 1.0f / (1.0f + expf(0.0f - in0__u1));
        float v0__u2 = 1.0f / (1.0f + expf(0.0f - in0__u2));
        float v0__u3 = 1.0f / (1.0f + expf(0.0f - in0__u3));
        float4 _vs_v0__u0_13811a1e = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&sigmoid[a0 * 4]) = _vs_v0__u0_13811a1e;
}
