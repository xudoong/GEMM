#include "dgemm.cuh"

const int TILE_M = 128;
const int TILE_N = 128;
const int TILE_K = 8;

const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

__global__ static void kernel_dgemm(DGEMM_FUNC_SIGNITURE) {
    __shared__ double A_shared[TILE_M * TILE_K];
    __shared__ double B_shared[TILE_N * TILE_K];
  
    int Ai = blockIdx.x * TILE_M;
    int Bj = blockIdx.y * TILE_N;

    double *Cp = C + Ai * N + Bj;
  
    const int M_local = TILE_M / BLOCK_DIM_X;
    const int N_local = TILE_N / BLOCK_DIM_Y;

    double C_rf[M_local * N_local] = {0};

    for (int kb = 0; kb < K; kb += TILE_K) {
        const double *Ap = A + Ai * K + kb;
        const double *Bp = B + kb * N + Bj;

        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        int nthreads = blockDim.x * blockDim.y;

        // load A tiles to shared memory
        for (int t = tid; t < TILE_M * TILE_K; t += nthreads) {
            int di = t % TILE_M;
            int dk = t / TILE_M;
            A_shared[t] = Ap[di * K + dk];
        }

        // load B tile to shared memory
        for (int t = tid; t < TILE_K * TILE_N; t += nthreads) {
            int dj = t % TILE_N;
            int dk = t / TILE_N;
            B_shared[t] = Bp[dk * N + dj];
        }

        __syncthreads();

        // compute
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < M_local; mi++) {
                for (int nj = 0; nj < N_local; nj++) {
                    int i = threadIdx.x + blockDim.x * mi;
                    int j = threadIdx.y + blockDim.y * nj;
                    C_rf[mi * N_local + nj] += A_shared[i + k * TILE_M] * B_shared[j + k * TILE_N];
                }
            }
        }
        __syncthreads();
    }

    // write to C
    for (int mi = 0; mi < M_local; mi++) {
        for (int nj = 0; nj < N_local; nj++) {
            int i = threadIdx.x + blockDim.x * mi;
            int j = threadIdx.y + blockDim.y * nj;
            Cp[i * N + j] = alpha * C_rf[mi * N_local + nj] + beta * Cp[i * N + j];
        }
    }
}

void dgemm_03_shmem(DGEMM_FUNC_SIGNITURE) {
    assert(M % TILE_M == 0);
    assert(N % TILE_N == 0);
    assert(K % TILE_K == 0);
    
    dim3 gridDim(CEIL_DIV(M, TILE_M), CEIL_DIV(N, TILE_N));
    dim3 blockDim(16, 16);
    kernel_dgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}