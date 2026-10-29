extern "C" __global__
__launch_bounds__(128) void k_layer_norm_42c036(const float* x, const float* layer_norm_mean_count, const float* layer_norm_var_count, const float* layer_norm_eps, const float* w, const float* b, float* layer_norm) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 512; a1 += 128) {
            float in3 = x[a0 * 512 + a1];
            acc0 += in3;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        float in1 = layer_norm_mean_count[0];
        float v0 = 1.0f / in1;
        float v2 = acc0 * v0;
        float acc1 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float in4 = x[a0 * 512 + a1];
            float v3 = in4 - v2;
            float v4 = v3 * v3;
            acc1 += v4;
        }
        float in2 = layer_norm_var_count[0];
        float v1 = 1.0f / in2;
        float v5 = acc1 * v1;
        float in0 = layer_norm_eps[0];
        float v6 = in0 + v5;
        float v7 = rsqrtf(v6);
        for (int a3 = a1_co; a3 < 512; a3 += 128) {
            float in5 = x[a0 * 512 + a3];
            float v8 = in5 - v2;
            float v9 = v7 * v8;
            float in7 = w[a3];
            float v10 = in7 * v9;
            float in6 = b[a3];
            float v11 = in6 + v10;
            layer_norm[a0 * 512 + a3] = v11;
        }
}
