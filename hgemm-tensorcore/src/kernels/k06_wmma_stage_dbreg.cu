#include "gemm.cuh"

const int bM = 256;
const int bN = 128;
const int bK = 32;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 16;
const int tK = 16;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

//const int num_warps = num_wM * num_wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

const int ldWidth = 128 / 8 / sizeof(half); // 8
const int bK2 = bK / ldWidth; // 4
const int bN2 = bN / ldWidth; // 16

const int nthreads = 32 * num_wM * num_wN;
const int stride_A_m = nthreads / bK2;
const int stride_B_n = nthreads / bK2;
const int stride_C_m = nthreads / bN2;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

#define sA(stage) (&sA[stage * bM * (bK + pad_half)])
#define sB(stage) (&sB[stage * bN * (bK + pad_half)])

const int pad_half = 8;
const int pad_float4 = 1;

constexpr int num_stages = 3;

using fragA_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tM, tN, tK, half, nvcuda::wmma::row_major>;
using fragB_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tM, tN, tK, half, nvcuda::wmma::col_major>;
using accum_t = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tM, tN, tK, half>;

__device__ static void load_shared_A(half *sA, const half *A, int K, int offset_A_m, int offset_A_k) {
    for (int i = 0; i < bM; i += stride_A_m) {
        void *dst       = &FLOAT4(sA)[(i + offset_A_m) * (bK2 + pad_float4) + offset_A_k];
        const void *src = &CFLOAT4(A)[(i + offset_A_m) * (K / ldWidth) + offset_A_k];
        copy_async<sizeof(float4)>(dst, src);
    }
}

__device__ static void load_shared_B(half *sB, const half *B, int K, int offset_B_n, int offset_B_k) {
    for (int j = 0; j < bN; j += stride_B_n) {
        void *dst       = &FLOAT4(sB)[(j + offset_B_n) * (bK2 + pad_float4) + offset_B_k];
        const void *src = &CFLOAT4(B)[(j + offset_B_n) * (K / ldWidth) + offset_B_k];
        copy_async<sizeof(float4)>(dst, src);
    }
}

__device__ static void store_shared_C(half *C, const half *sC, int N, int offset_C_m, int offset_C_n) {
    for (int i = 0; i < bM; i += stride_C_m) {
        float4 *dst       =   &FLOAT4(C)[(i + offset_C_m) * (N / ldWidth) + offset_C_n];
        const float4 *src = &CFLOAT4(sC)[(i + offset_C_m) * (bN2 + pad_float4) + offset_C_n];
        *dst = *src;
    }
}

__device__ static void load_register(int wi, int wj, fragA_t *fragA, fragB_t *fragB, const half *sA, const half *sB, int k) {
    const half *sAw = &sA[wi * wM * (bK + pad_half)];
    const half *sBw = &sB[wj * wN * (bK + pad_half)];

    for (int i = 0; i < wM; i += tM) {
        nvcuda::wmma::load_matrix_sync(fragA[i / tM], &sAw[i * (bK + pad_half) + k], bK + pad_half);
    }
    for (int j = 0; j < wN; j += tN) {
        nvcuda::wmma::load_matrix_sync(fragB[j / tN], &sBw[j * (bK + pad_half) + k], bK + pad_half);
    }
}

__device__ static void do_wmma(fragA_t *fragA, fragB_t *fragB, accum_t *accum) {
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            auto &acc = accum[i / tM * num_tN + j / tN];
            nvcuda::wmma::mma_sync(acc, fragA[i / tM], fragB[j / tN], acc);
        }
    }
}

template <int stage_idx>
__device__ static void post_stage(const int wi, const int wj, int K, const half *sA, const half *sB, fragA_t fragA[][num_tM], fragB_t fragB[][num_tN], accum_t *accum, int &reg_st, int &reg_ld) {
    if (stage_idx > num_stages) {
        return;
    }

    int stage = (K / bK + stage_idx - 1) % num_stages;
    for (int k = tK; k < bK; k += tK) {
        load_register(wi, wj, fragA[reg_st], fragB[reg_st], sA(stage), sB(stage), k);
        do_wmma(fragA[reg_ld], fragB[reg_ld], accum);
        reg_st ^= 1;
        reg_ld ^= 1;
    }
    if (num_stages > stage_idx) {
        // prefetch the first tile of the next stage
        copy_async_wait_and_syncthreads<num_stages - 1 - stage_idx>();
        int next_stage = (stage + 1) % num_stages;
        load_register(wi, wj, fragA[reg_st], fragB[reg_st], sA(next_stage), sB(next_stage), 0);
    }
    // compute the last tile of this stage
    do_wmma(fragA[reg_ld], fragB[reg_ld], accum);
    reg_st ^= 1;
    reg_ld ^= 1;
}

__global__ static void kernel06(GEMM_FUNC_SIGNITURE) {
    extern __shared__ half smem_base[];
    half *sA = smem_base;
    half *sB = &sA[num_stages * bM * (bK + pad_half)];

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN * K;
    C += bi * bM * N + bj * bN;

    fragA_t fragA[2][num_tM];
    fragB_t fragB[2][num_tN];
    accum_t accum[num_tM * num_tN];

    for (int i = 0; i < num_tM * num_tN; i++) {
        nvcuda::wmma::fill_fragment(accum[i], 0.0);
    }

    const int wid = threadIdx.x / 32;
    const int wi = wid % num_wM;
    const int wj = wid / num_wM;

    const int tid = threadIdx.x;
    const int offset_A_m = tid / bK2;
    const int offset_A_k = tid % bK2;
    const int offset_B_n = tid / bK2;
    const int offset_B_k = tid % bK2;

    // pre-stage
    for (int stage = 0; stage < num_stages - 1; stage++) {
        load_shared_A(sA(stage), A + stage * bK, K, offset_A_m, offset_A_k);
        load_shared_B(sB(stage), B + stage * bK, K, offset_B_n, offset_B_k);
        copy_async_commit();
    }
    copy_async_wait_and_syncthreads<num_stages - 2>();

    int reg_st = 0;
    int reg_ld = 1;
    load_register(wi, wj, fragA[reg_st], fragB[reg_st], sA(0), sB(0), 0);
    reg_st ^= 1;
    reg_ld ^= 1;

    // main loop
    for (int stage = num_stages - 1; stage < K / bK; stage++) {
        int comp_stage = (stage - num_stages + 1) % num_stages;
        int copy_stage = stage % num_stages;

        for (int k = tK; k < bK; k += tK) {
            load_register(wi, wj, fragA[reg_st], fragB[reg_st], sA(comp_stage), sB(comp_stage), k);
            do_wmma(fragA[reg_ld], fragB[reg_ld], accum);
            reg_st ^= 1;
            reg_ld ^= 1;
        }

        load_shared_A(sA(copy_stage), A + stage * bK, K, offset_A_m, offset_A_k);
        load_shared_B(sB(copy_stage), B + stage * bK, K, offset_B_n, offset_B_k);
        copy_async_commit();
        copy_async_wait_and_syncthreads<num_stages - 2>();

        // prefetch registers for the next compute stage and compute the last tile in this compute stage
        int next_comp_stage = (comp_stage + 1) % num_stages;
        load_register(wi, wj, fragA[reg_st], fragB[reg_st], sA(next_comp_stage), sB(next_comp_stage), 0);
        do_wmma(fragA[reg_ld], fragB[reg_ld], accum);
        reg_st ^= 1;
        reg_ld ^= 1;
    }

    // compute remaining stages
    post_stage<2>(wi, wj, K, sA, sB, fragA, fragB, accum, reg_st, reg_ld);
    post_stage<3>(wi, wj, K, sA, sB, fragA, fragB, accum, reg_st, reg_ld);
    post_stage<4>(wi, wj, K, sA, sB, fragA, fragB, accum, reg_st, reg_ld);

    // write back result
    half *sC = smem_base;
    half *sCw = sC + wi * wM * (bN + pad_half) + wj * wN;
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            auto &acc = accum[i / tM * num_tN + j / tN];
            nvcuda::wmma::store_matrix_sync(&sCw[i * (bN + pad_half) + j], acc, bN + pad_half, nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();
    const int offset_C_m = tid / bN2;
    const int offset_C_n = tid % bN2;
    store_shared_C(C, sC, N, offset_C_m, offset_C_n);
}

void gemm_06_wmma_stage_dbreg(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    static_assert(1 < num_stages && num_stages <= 4);
    if ((float)alpha != 1 || (float)beta != 0) {
        std::cout << "gemm_06_wmma_stage_dbreg kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    // set dynamic shared memory
    size_t smem_size = std::max(num_stages * (bM + bN) * (bK + pad_half),
                                bM * (bN + pad_half)) * sizeof(half);
    static bool printed = false;
    if (!printed) {
        printed = true;
        std::cout << "dynamic shared memory: " << float(smem_size) / 1024 << " KB" << std::endl;
    }
    cudaFuncSetAttribute(kernel06,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32 * num_wM * num_wN);
    kernel06<<<gridDim, blockDim, smem_size>>>(GEMM_FUNC_PARAM);
}