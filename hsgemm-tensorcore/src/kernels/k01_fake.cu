#include "gemm.cuh"

const int ctaM = 32;
const int ctaN = 32;

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__global__ static void kernel(GEMM_FUNC_SIGNITURE) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragA;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragB;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> accum;

    for (int k = 0; k < K; k += wmmaK) {
        nvcuda::wmma::mma_sync(accum, fragA, fragB, accum);
    }
    nvcuda::wmma::store_matrix_sync(C, accum, N, nvcuda::wmma::mem_row_major);
}

void gemm_01_fake(GEMM_FUNC_SIGNITURE) {
    dim3 gridDim(CEIL_DIV(M, ctaM), CEIL_DIV(N, ctaN));
    dim3 blockDim(32, ctaM / wmmaM, ctaN / wmmaN);
    kernel<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}
