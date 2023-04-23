#include "dgemm.cuh"


__global__ static void kernel_dgemm_naive(DGEMM_FUNC_SIGNITURE) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        double tmp = 0.0;
        for (int k = 0; k < K; k++) {
            tmp += A[x * K + k] * B[k * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void dgemm_01_naive(DGEMM_FUNC_SIGNITURE) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    kernel_dgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}