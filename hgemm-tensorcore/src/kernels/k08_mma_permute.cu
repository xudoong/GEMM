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
const int ldWidth = 128 / 8 / sizeof(half); // 8
const int bK2 = bK / ldWidth;
const int bN2 = bN / ldWidth;
const int smem_stride_ij = nthreads / bK2;
const int stride_C_m = nthreads / bN2;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

const int pad_half = 8;
const int pad_float4 = 1;

__device__ static void load_shared_AB(half *shared_ptr, const half *global_ptr, int K, int tid, int count, int offset_ij, int offset_k) {
    for (int i = 0; i < count; i += smem_stride_ij) {
        int row = (i * bK2 + tid) / 8;
        int col = (i * bK2 + tid) % 8;
        col = col ^ (row % bK2);

        float4 *dst       = &FLOAT4(shared_ptr)[row * 8 + col];
        const float4 *src = &CFLOAT4(global_ptr)[(i + offset_ij) * (K / ldWidth) + offset_k];
        *dst = *src;
    }
}

__device__ static void store_shared_C(half *C, const half *sC, int N, int tid, int offset_C_m, int offset_C_n) {
    for (int i = 0; i < bM; i += stride_C_m) {
        float4 *dst       =   &FLOAT4(C)[(i + offset_C_m) * (N / ldWidth) + offset_C_n];
        const float4 *src = &CFLOAT4(sC)[(i + offset_C_m) * (bN2 + pad_float4) + offset_C_n];
        *dst = *src;
    }
}

__device__ static void compute(const int lane, const int wi, const int wj, const half *sA, const half *sB,
                               uint32_t *rA, uint32_t *rB, uint32_t *rC) {
    const half *sAw = &sA[wi * wM * bK];
    const half *sBw = &sB[wj * wN * bK];

    for (int k = 0; k < bK; k += tK) {
        for (int i = 0; i < wM; i += tM) {
            int idx = (i + lane % tM) * bK2 + (k / 8 + lane / tM);
            int row = idx / 8;
            int col = idx % 8;
            col = col ^ (row % bK2);
            ldmatrixA(&FLOAT4(sAw)[row * 8 + col], &rA[i / tM * 4]);
        }

        for (int j = 0; j < wN; j += tN) {
            int idx = (j + lane % tN) * bK2 + (k / 8 + lane / tN);
            int row = idx / 8;
            int col = idx % 8;
            col = col ^ (row % bK2);
            ldmatrixB(&FLOAT4(sBw)[row * 8 + col], &rB[j / tN * 2]);
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

__global__ static void kernel08(GEMM_FUNC_SIGNITURE) {
    extern __shared__ half smem_base[];
    half *sA = smem_base;
    half *sB = &sA[bM * bK];

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

    const int offset_ij = tid / bK2;
    const int offset_k = tid % bK2;

    for (int stage = 0; stage < K / bK; stage++) {
        load_shared_AB(sA, A + stage * bK, K, tid, bM, offset_ij, offset_k);
        load_shared_AB(sB, B + stage * bK, K, tid, bN, offset_ij, offset_k);
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
    store_shared_C(C, sC, N, tid, offset_C_m, offset_C_n);
}

void gemm_08_mma_permute(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    if ((float)alpha != 1 || (float)beta != 0) {
        std::cout << "gemm_08_mma_permute kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    size_t smem_size = std::max((bM + bN) * bK, bM * (bN + pad_half)) * sizeof(half);
    static bool printed = false;
    if (!printed) {
        printed = true;
        std::cout << "dynamic shared memory: " << float(smem_size) / 1024 << " KB" << std::endl;
    }

    cudaFuncSetAttribute(kernel08,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32 * num_wM * num_wN);
    kernel08<<<gridDim, blockDim, smem_size>>>(GEMM_FUNC_PARAM);
}