#include "gemm.cuh"

const int bM = 128;
const int bN = 128;
const int bK = 32;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 16;
const int tK = 16;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

__global__ static void kernel(GEMM_FUNC_SIGNITURE) {
    __shared__ half sA[bM * bK];
    __shared__ half sB[bK * bN];

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
    half *sAw = &sA[wi * wM * bK];
    half *sBw = &sB[wj * wN];

    const int stride_A_m = nthreads / bK;
    const int offset_A_m = tid / bK;
    const int offset_A_k = tid % bK;
    const int stride_B_k = nthreads / bN;
    const int offset_B_k = tid / bN;
    const int offset_B_n = tid % bN;

    for (int _k = 0; _k < K; _k += bK) {
        // load global mem -> shared mem
        for (int i = 0; i < bM; i += stride_A_m) {
            sA[(i + offset_A_m) * bK + offset_A_k] = A[(i + offset_A_m) * K + offset_A_k];
        }
        for (int i = 0; i < bK; i += stride_B_k) {
            sB[(i + offset_B_k) * bN + offset_B_n] = B[(i + offset_B_k) * N + offset_B_n];
        }
        __syncthreads();

        // compute
        for (int k = 0; k < bK; k += tK) {
            for (int j = 0; j < wN; j += tN) {
                nvcuda::wmma::load_matrix_sync(fragB[j / tN], &sBw[k * bN + j], bN);
            }
            for (int i = 0; i < wM; i += tM) {
                nvcuda::wmma::load_matrix_sync(fragA, &sAw[i * bK + k], bK);
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

void gemm_03_wmma_shmem(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    if (alpha != 1 || beta != 0) {
        std::cout << "gemm_03_wmma_shmem kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32, num_wM, num_wN);
    kernel<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}