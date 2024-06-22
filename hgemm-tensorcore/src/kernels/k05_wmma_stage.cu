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

__device__ static void compute(int wi, int wj, const half *sA, const half *sB, fragA_t *fragA, fragB_t *fragB, accum_t *accum) {

    const half *sAw = &sA[wi * wM * (bK + pad_half)];
    const half *sBw = &sB[wj * wN * (bK + pad_half)];
    for (int k = 0; k < bK; k += tK) {
        for (int i = 0; i < wM; i += tM) {
            nvcuda::wmma::load_matrix_sync(fragA[i / tM], &sAw[i * (bK + pad_half) + k], bK + pad_half);
        }
        for (int j = 0; j < wN; j += tN) {
            nvcuda::wmma::load_matrix_sync(fragB[j / tN], &sBw[j * (bK + pad_half) + k], bK + pad_half);
        }
        for (int i = 0; i < wM; i += tM) {
            for (int j = 0; j < wN; j += tN) {
                auto &acc = accum[i / tM * num_tN + j / tN];
                nvcuda::wmma::mma_sync(acc, fragA[i / tM], fragB[j / tN], acc);
            }
        }
    }
}

__global__ static void kernel05(GEMM_FUNC_SIGNITURE) {
    extern __shared__ half smem_base[];
    half *sA = smem_base;
    half *sB = &sA[num_stages * bM * (bK + pad_half)];
#define sA(stage) (&sA[stage * bM * (bK + pad_half)])
#define sB(stage) (&sB[stage * bN * (bK + pad_half)])

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN * K;
    C += bi * bM * N + bj * bN;

    fragA_t fragA[num_tM];
    fragB_t fragB[num_tN];
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

    for (int stage = 0; stage < num_stages - 1; stage++) {
        load_shared_A(sA(stage), A + stage * bK, K, offset_A_m, offset_A_k);
        load_shared_B(sB(stage), B + stage * bK, K, offset_B_n, offset_B_k);
        copy_async_commit();
    }

    copy_async_wait_and_syncthreads<num_stages - 2>();

    for (int stage = num_stages - 1; stage < K / bK; stage++) {
        int comp_stage = (stage - num_stages + 1) % num_stages;
        int copy_stage = stage % num_stages;
        load_shared_A(sA(copy_stage), A + stage * bK, K, offset_A_m, offset_A_k);
        compute(wi, wj, sA(comp_stage), sB(comp_stage), fragA, fragB, accum);
        load_shared_B(sB(copy_stage), B + stage * bK, K, offset_B_n, offset_B_k);
        copy_async_commit();
        copy_async_wait_and_syncthreads<num_stages - 2>();
    }

    // compute remaining stages
    if (num_stages >= 2) {
        int stage = (K / bK + 1) % num_stages;
        compute(wi, wj, sA(stage), sB(stage), fragA, fragB, accum);
    }
    if (num_stages >= 3) {
        int stage = (K / bK + 2) % num_stages;
        copy_async_wait_and_syncthreads<num_stages - 3>();
        compute(wi, wj, sA(stage), sB(stage), fragA, fragB, accum);
    }
    if (num_stages >= 4) {
        int stage = (K / bK + 3) % num_stages;
        copy_async_wait_and_syncthreads<num_stages - 4>();
        compute(wi, wj, sA(stage), sB(stage), fragA, fragB, accum);
    }

    // write to C
    // version 1: directly write to global memory
//    C += wi * wM * N + wj * wN;
//    for (int i = 0; i < wM; i += tM) {
//        for (int j = 0; j < wN; j += tN) {
//            auto &acc = accum[i / tM * num_tN + j / tN];
//            nvcuda::wmma::store_matrix_sync(&C[i * N + j], acc, N, nvcuda::wmma::mem_row_major);
//        }
//    }
    // version 2: store to shared memory then to global memory
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

void gemm_05_wmma_stage(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    static_assert(1 < num_stages && num_stages <= 4);
    if ((float)alpha != 1 || (float)beta != 0) {
        std::cout << "gemm_05_wmma_stage kernel only supports computing C=A*B (alpha=1, beta=0)\n";
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
    cudaFuncSetAttribute(kernel05,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32 * num_wM * num_wN);
    kernel05<<<gridDim, blockDim, smem_size>>>(GEMM_FUNC_PARAM);
}