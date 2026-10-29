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
__launch_bounds__(64) void k_matmul_d4bced(const float* x, const float* y, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 64;
        int a1_b = (_gid / 64) % 1;
        int a0_u = (_gid / 64) % 1;
        int a1_u = (_gid) % 64;
        float acc0__c0_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[64];
        __shared__ __align__(16) float _b_smem[4096];
        for (int _ks = 0; _ks < 64; _ks += 64) {
            for (int _fa = a0_u * 64 + a1_u; _fa < 16; _fa += 64) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x[(a0_b + _fa / 16) * 64 + (_ks + 4 * (_fa % 16))]);
            }
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + (a0_u * 64 + a1_u) / 16) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[256 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 4)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[512 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 8)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[768 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 12)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 16)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[1280 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 20)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[1536 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 24)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[1792 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 28)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 32)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[2304 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 36)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[2560 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 40)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[2816 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 44)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[3072 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 48)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[3328 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 52)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[3584 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 56)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[3840 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &y[(_ks + ((a0_u * 64 + a1_u) / 16 + 60)) * 64 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 64; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u - a1_b * 64)];
                float in0__c0_0 = _a_smem[(a0_b + a0_u - a0_b) * 64 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
            }
            __syncthreads();
        }
        matmul[(a0_b + a0_u) * 64 + (a1_b * 64 + a1_u)] = acc0__c0_0;
}
