#include "gemm.cuh"

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__global__ static void kernel02(GEMM_FUNC_SIGNITURE) {
    const int bi = blockIdx.x * wmmaM;
    const int bj = blockIdx.y * wmmaN;

    A += bi * K;
    B += bj * K;
    C += bi * N + bj;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragA;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> fragB;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> accum;
    nvcuda::wmma::fill_fragment(accum, 0.0);

    for (int k = 0; k < K; k += wmmaK) {
        nvcuda::wmma::load_matrix_sync(fragA, A, K);
        nvcuda::wmma::load_matrix_sync(fragB, B, K);
        nvcuda::wmma::mma_sync(accum, fragA, fragB, accum);
        A += wmmaK;
        B += wmmaK;
    }
    nvcuda::wmma::store_matrix_sync(C, accum, N, nvcuda::wmma::mem_row_major);
}

void gemm_02_naive(GEMM_FUNC_SIGNITURE) {
    assert(M % wmmaM == 0);
    assert(N % wmmaN == 0);
    dim3 gridDim(CEIL_DIV(M, wmmaM), CEIL_DIV(N, wmmaN));
    dim3 blockDim(32);
    kernel02<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}
