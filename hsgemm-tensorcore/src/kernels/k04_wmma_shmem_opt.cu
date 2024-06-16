#include "gemm.cuh"

const int bM = 128;
const int bN = 128;
const int bK = 64;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 16;
const int tK = 16;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

const int ldWidth = 128 / 8 / sizeof(half);

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

__global__ static void kernel(GEMM_FUNC_SIGNITURE) {
    __shared__ half sA[bM * (bK + 8)];
    __shared__ half sB[bK * (bN + 8)];

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN;
    C += bi * bM * N + bj * bN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tM, tN, tK, half, nvcuda::wmma::row_major> fragA;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tM, tN, tK, half, nvcuda::wmma::row_major> fragB[num_tN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tM, tN, tK, float> accum[num_tM * num_tN];
    for (int i = 0; i < num_tM * num_tN; i++) {
        nvcuda::wmma::fill_fragment(accum[i], 0.0);
    }

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    const int wi = threadIdx.y;
    const int wj = threadIdx.z;

    C += wi * wM * N + wj * wN;
    half *sAw = &sA[wi * wM * (bK + 8)];
    half *sBw = &sB[wj * wN];

    const int bK2 = bK / ldWidth;
    const int bN2 = bN / ldWidth;
    const int stride_A_m = nthreads / bK2;
    const int offset_A_m = tid / bK2;
    const int offset_A_k = tid % bK2;
    const int stride_B_k = nthreads / bN2;
    const int offset_B_k = tid / bN2;
    const int offset_B_n = tid % bN2;

    for (int _k = 0; _k < K; _k += bK) {
        // load global mem -> shared mem
        for (int i = 0; i < bM; i += stride_A_m) {
            FLOAT4(sA)[(i + offset_A_m) * (bK2 + 1) + offset_A_k] = CFLOAT4(A)[(i + offset_A_m) * (K / ldWidth) + offset_A_k];
        }
        for (int i = 0; i < bK; i += stride_B_k) {
            FLOAT4(sB)[(i + offset_B_k) * (bN2 + 1) + offset_B_n] = CFLOAT4(B)[(i + offset_B_k) * (N / ldWidth) + offset_B_n];
        }
        __syncthreads();

        // compute
        for (int k = 0; k < bK; k += tK) {
            for (int j = 0; j < wN; j += tN) {
                nvcuda::wmma::load_matrix_sync(fragB[j / tN], &sBw[k * (bN + 8) + j], bN + 8);
            }
            for (int i = 0; i < wM; i += tM) {
                nvcuda::wmma::load_matrix_sync(fragA, &sAw[i * (bK + 8) + k], bK + 8);
                for (int j = 0; j < wN; j += tN) {
                    auto &acc = accum[i / tM * num_tN + j / tN];
                    nvcuda::wmma::mma_sync(acc, fragA, fragB[j / tN], acc);
                }
            }
        }
        __syncthreads();

        // advance to next tile
        A += bK;
        B += bK * N;
    }

    // write to C
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            auto &acc = accum[i / tM * num_tN + j / tN];
            nvcuda::wmma::store_matrix_sync(&C[i * N + j], acc, N, nvcuda::wmma::mem_row_major);
        }
    }
}

void gemm_04_wmma_shmem_opt(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    if (alpha != 1 || beta != 0) {
        std::cout << "gemm_04_wmma_shmem_opt kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32, num_wM, num_wN);
    kernel<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}