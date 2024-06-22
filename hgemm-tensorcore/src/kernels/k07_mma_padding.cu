#include "gemm.cuh"

const int bM = 128;
const int bN = 128;
const int bK = 64;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 8;
const int tK = 16;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

const int nthreads = 32 * num_wM * num_wN;

// use for load global memory -> shared memory
const int ldWidth = 128 / 8 / sizeof(half);
const int bK2 = bK / ldWidth;
const int bN2 = bN / ldWidth;
const int stride_A_m = nthreads / bK2;
const int stride_B_n = nthreads / bK2;
const int stride_C_m = nthreads / bN2;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

// padding shared memory to avoid bank conflict
const int pad_half = 8;
const int pad_float4 = 1;

__device__ static void load_shared_A(half *sA, const half *A, int K, int offset_A_m, int offset_A_k) {
    for (int i = 0; i < bM; i += stride_A_m) {
        float4 *dst       = &FLOAT4(sA)[(i + offset_A_m) * (bK2 + pad_float4) + offset_A_k];
        const float4 *src = &CFLOAT4(A)[(i + offset_A_m) * (K / ldWidth) + offset_A_k];
        *dst = *src;
    }
}

__device__ static void load_shared_B(half *sB, const half *B, int K, int offset_B_n, int offset_B_k) {
    for (int j = 0; j < bN; j += stride_B_n) {
        float4 *dst       = &FLOAT4(sB)[(j + offset_B_n) * (bK2 + pad_float4) + offset_B_k];
        const float4 *src = &CFLOAT4(B)[(j + offset_B_n) * (K / ldWidth) + offset_B_k];
        *dst = *src;
    }
}

__device__ static void store_shared_C(half *C, const half *sC, int N, int offset_C_m, int offset_C_n) {
    for (int i = 0; i < bM; i += stride_C_m) {
        float4 *dst       =   &FLOAT4(C)[(i + offset_C_m) * (N / ldWidth) + offset_C_n];
        const float4 *src = &CFLOAT4(sC)[(i + offset_C_m) * (bN2 + pad_float4) + offset_C_n];
        *dst = *src;
    }
}

__device__ static void compute(const int lane, const int wi, const int wj, const half *sA, const half *sB,
                               uint32_t *rA, uint32_t *rB, uint32_t *rC) {
    const int tAi = lane % 16;
    const int tAk = lane / 16;
    const int tBj = lane % 8;
    const int tBk = lane / 8;

    const half *sAw = &sA[wi * wM * (bK + pad_half)];
    const half *sBw = &sB[wj * wN * (bK + pad_half)];

    for (int k = 0; k < bK; k += tK) {
        for (int i = 0; i < wM; i += tM) {
            ldmatrixA(&sAw[(i + tAi) * (bK + pad_half) + k + tAk * 8], &rA[i / tM * 4]);
        }

        for (int j = 0; j < wN; j += tN) {
            ldmatrixB(&sBw[(j + tBj) * (bK + pad_half) + k + tBk * 8], &rB[j / tN * 2]);
        }
        for (int i = 0; i < wM; i += tM) {
            for (int j = 0; j < wN; j += tN) {
                mma_m16n8k16(&rA[i / tM * 4],
                             &rB[j / tN * 2],
                             &rC[(i / tM * num_tN + j / tN) * 2]);
            }
        }
    }
}

__global__ static void kernel07(GEMM_FUNC_SIGNITURE) {
    extern __shared__ half smem_base[];
    half *sA = smem_base;
    half *sB = &sA[bM * (bK + pad_half)];

    // m16n8k16 mma
    uint32_t rA[num_tM * 4];
    uint32_t rB[num_tN * 2];
    uint32_t rC[num_tM * num_tN * 2] = {0};

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN * K;
    C += bi * bM * N + bj * bN;

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int wi = wid % num_wM;
    const int wj = wid / num_wM;

    const int offset_A_m = tid / bK2;
    const int offset_A_k = tid % bK2;
    const int offset_B_n = tid / bK2;
    const int offset_B_k = tid % bK2;

    for (int stage = 0; stage < K / bK; stage++) {
        load_shared_A(sA, A + stage * bK, K, offset_A_m, offset_A_k);
        load_shared_B(sB, B + stage * bK, K, offset_B_n, offset_B_k);
        __syncthreads();

        compute(lane, wi, wj, sA, sB, rA, rB, rC);
        __syncthreads();
    }

    // write back to C

    const int tCi = lane / 4;
    const int tCj = lane % 4;
    half *sC = smem_base;
    half *sCw = sC + wi * wM * (bN + pad_half) + wj * wN;
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            uint32_t *rCt = &rC[(i / tM * num_tN + j / tN) * 2];
            *(uint32_t *)&sCw[(i + tCi + 0) * (bN + pad_half) + j + tCj * 2] = rCt[0];
            *(uint32_t *)&sCw[(i + tCi + 8) * (bN + pad_half) + j + tCj * 2] = rCt[1];
        }
    }
    __syncthreads();
    const int offset_C_m = tid / bN2;
    const int offset_C_n = tid % bN2;
    store_shared_C(C, sC, N, offset_C_m, offset_C_n);
}

void gemm_07_mma_padding(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    if ((float)alpha != 1 || (float)beta != 0) {
        std::cout << "gemm_07_mma_padding kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    size_t smem_size = std::max((bM + bN) * (bK + pad_half),
                                bM * (bN + pad_half)) * sizeof(half);
    static bool printed = false;
    if (!printed) {
        printed = true;
        std::cout << "dynamic shared memory: " << float(smem_size) / 1024 << " KB" << std::endl;
    }

    cudaFuncSetAttribute(kernel07,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32 * num_wM * num_wN);
    kernel07<<<gridDim, blockDim, smem_size>>>(GEMM_FUNC_PARAM);
}