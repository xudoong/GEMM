#include "dgemm.cuh"


__global__ static void kernel_dgemm_ideal(DGEMM_FUNC_SIGNITURE) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    double tmp = 0.0;
    for (int k = 0; k < K; k++) {
      tmp += A[0] * B[0];
    }
    C[0] = alpha * tmp + beta * C[0];
  }
}

void dgemm_02_ideal(DGEMM_FUNC_SIGNITURE) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  kernel_dgemm_ideal<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}