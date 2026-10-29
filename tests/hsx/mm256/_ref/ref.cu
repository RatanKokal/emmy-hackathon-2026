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
__launch_bounds__(512) void (const float* x, const float* y, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 3072;
        int a0_b = (_gid / 1024) % 3;
        int a1_b = (_gid / 512) % 2;
        int a0_u = (_gid / 64) % 8;
        int a1_u = (_gid) % 64;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c8_0 = 0.0f;
        float acc0__c9_0 = 0.0f;
        float acc0__c10_0 = 0.0f;
        float acc0__c11_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[384];
        __shared__ __align__(16) float _b_smem[256];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a0_u * 64 + a1_u; _fa < 96; _fa += 512) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x[((a0_b * 96 + _fa < 256) ? (a0_b * 96 + _fa) : (255)) * 256 + (_a2_ks * 4 + _ks)]);
            }
            for (int _fb = a0_u * 64 + a1_u; _fb < 64; _fb += 512) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &y[(_a2_ks * 4 + (_ks + _fb / 16)) * 128 + (a1_b * 64 + 4 * (_fb % 16))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u - a1_b * 64)];
                float in0__c0_0 = _a_smem[(a0_b * 96 + a0_u * 12 - a0_b * 96) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 1 - a0_b * 96) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float in0__c2_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 2 - a0_b * 96) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float in0__c3_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 3 - a0_b * 96) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float in0__c4_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 4 - a0_b * 96) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float in0__c5_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 5 - a0_b * 96) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float in0__c6_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 6 - a0_b * 96) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float in0__c7_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 7 - a0_b * 96) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                float in0__c8_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 8 - a0_b * 96) * 4 + _ki];
                float acc0__v__c8_0 = in1__c0_0 * in0__c8_0;
                acc0__c8_0 += acc0__v__c8_0;
                float in0__c9_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 9 - a0_b * 96) * 4 + _ki];
                float acc0__v__c9_0 = in1__c0_0 * in0__c9_0;
                acc0__c9_0 += acc0__v__c9_0;
                float in0__c10_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 10 - a0_b * 96) * 4 + _ki];
                float acc0__v__c10_0 = in1__c0_0 * in0__c10_0;
                acc0__c10_0 += acc0__v__c10_0;
                float in0__c11_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 11 - a0_b * 96) * 4 + _ki];
                float acc0__v__c11_0 = in1__c0_0 * in0__c11_0;
                acc0__c11_0 += acc0__v__c11_0;
            }
            __syncthreads();
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c0_0);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 1) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c1_0);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 2) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c2_0);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 3) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c3_0);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 4) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c4_0);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 5) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c5_0);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 6) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c6_0);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 7) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c7_0);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 8) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c8_0);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 9) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c9_0);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 10) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c10_0);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 256) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 11) % 256 * 128 + (a1_b * 64 + a1_u)], acc0__c11_0);
        }
}
