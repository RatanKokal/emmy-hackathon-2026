static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(128) void k_matmul_288631(const float* x, const float* y, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a1_b = (_gid / 128) % 4;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c4_1 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c5_1 = 0.0f;
        __shared__ __align__(16) float _a_smem[6144];
        __shared__ __align__(16) float _b_smem[4096];
        for (int _ks = 0; _ks < 128; _ks += 128) {
            emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + (a0_u * 16 + a1_u) / 32 < 128) ? (a0_b * 48 + (a0_u * 16 + a1_u) / 32) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[512 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 4) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 4)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 8) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 8)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 12) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 12)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[2048 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 16) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 16)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[2560 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 20) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 20)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[3072 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 24) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 24)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[3584 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 28) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 28)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[4096 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 32) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 32)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[4608 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 36) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 36)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[5120 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 40) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 40)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[5632 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[((a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 44) < 128) ? (a0_b * 48 + ((a0_u * 16 + a1_u) / 32 + 44)) : (127)) * 128 + (_ks + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + (a0_u * 16 + a1_u) / 8) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[512 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 16)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 32)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 48)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 64)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[2560 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 80)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[3072 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 96)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[3584 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &y[(_ks + ((a0_u * 16 + a1_u) / 8 + 112)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            for (int _ki = 0; _ki < 128; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u * 2 - a1_b * 32)];
                float in0__c0_0 = _a_smem[(a0_b * 48 + a0_u * 6 - a0_b * 48) * 128 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u * 2 + 1 - a1_b * 32)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(a0_b * 48 + a0_u * 6 + 1 - a0_b * 48) * 128 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(a0_b * 48 + a0_u * 6 + 2 - a0_b * 48) * 128 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(a0_b * 48 + a0_u * 6 + 3 - a0_b * 48) * 128 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                float in0__c4_0 = _a_smem[(a0_b * 48 + a0_u * 6 + 4 - a0_b * 48) * 128 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                float in0__c5_0 = _a_smem[(a0_b * 48 + a0_u * 6 + 5 - a0_b * 48) * 128 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
            }
            __syncthreads();
        }
        if (a0_b * 48 + a0_u * 6 + 0 < 128) {
            matmul[(a0_b * 48 + a0_u * 6) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c0_0;
        }
        if (a0_b * 48 + a0_u * 6 + 0 < 128) {
            matmul[(a0_b * 48 + a0_u * 6) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c0_1;
        }
        if (a0_b * 48 + a0_u * 6 + 1 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 1) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c1_0;
        }
        if (a0_b * 48 + a0_u * 6 + 1 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 1) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c1_1;
        }
        if (a0_b * 48 + a0_u * 6 + 2 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 2) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c2_0;
        }
        if (a0_b * 48 + a0_u * 6 + 2 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 2) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c2_1;
        }
        if (a0_b * 48 + a0_u * 6 + 3 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 3) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c3_0;
        }
        if (a0_b * 48 + a0_u * 6 + 3 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 3) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c3_1;
        }
        if (a0_b * 48 + a0_u * 6 + 4 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 4) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c4_0;
        }
        if (a0_b * 48 + a0_u * 6 + 4 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 4) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c4_1;
        }
        if (a0_b * 48 + a0_u * 6 + 5 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 5) % 128 * 128 + (a1_b * 32 + a1_u * 2)] = acc0__c5_0;
        }
        if (a0_b * 48 + a0_u * 6 + 5 < 128) {
            matmul[(a0_b * 48 + a0_u * 6 + 5) % 128 * 128 + (a1_b * 32 + a1_u * 2 + 1)] = acc0__c5_1;
        }
}
