#include "dgemm.cuh"

const int TILE_M = 128;
const int TILE_N = 128;
const int TILE_K = 16;

constexpr int BLOCK_DIM_X = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WRAPS_M = 2;
constexpr int NUM_WRAPS_N = 4;
constexpr int WARP_DIM_M = 8;
constexpr int WARP_DIM_N = 4;

constexpr int WARP_TILE_M = TILE_M / NUM_WRAPS_M;
constexpr int WARP_TILE_N = TILE_N / NUM_WRAPS_N;

static_assert (BLOCK_DIM_X == WARP_SIZE * NUM_WRAPS_M * NUM_WRAPS_N);


__global__ static void kernel_dgemm(DGEMM_FUNC_SIGNITURE) {
    __shared__ double A_shared[TILE_M * TILE_K];
    __shared__ double B_shared[TILE_N * TILE_K];
  
    int Ai = blockIdx.x * TILE_M;
    int Bj = blockIdx.y * TILE_N;

    double *Cp = C + Ai * N + Bj;
  
    constexpr int M_local = TILE_M / NUM_WRAPS_M / WARP_DIM_M;
    constexpr int N_local = TILE_N / NUM_WRAPS_N / WARP_DIM_N;

    double C_rf[M_local * N_local] = {0};
    double A_rf[M_local];
    double B_rf[N_local];

    const int tid = threadIdx.x;
    const int nthreads = BLOCK_DIM_X;
    const int warp_id = tid / WARP_SIZE;
    const int warp_offset = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WRAPS_N;
    const int warp_id_n = warp_id % NUM_WRAPS_N;
    const int warp_offset_m = warp_offset / WARP_DIM_N;
    const int warp_offset_n = warp_offset % WARP_DIM_N;

    const int Aii = Ai + warp_id_m * WARP_TILE_M;
    const int Bjj = Bj + warp_id_n * WARP_TILE_N;
    double *Cpp = C + Aii * N + Bjj;

    for (int kb = 0; kb < K; kb += TILE_K) {
        const double *Ap = A + Ai * K + kb;
        const double *Bp = B + kb * N + Bj;

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
                    int i = warp_offset_m + WARP_DIM_M * mi;
                    int j = warp_offset_n + WARP_DIM_N * nj;
                    C_rf[mi * N_local + nj] += A_shared[i + warp_id_m * WARP_TILE_M + k * TILE_M] * B_shared[j + warp_id_n * WARP_TILE_N + k * TILE_N];
                }
            }
        }
        __syncthreads();
    }

    // write to C
    for (int mi = 0; mi < M_local; mi++) {
        for (int nj = 0; nj < N_local; nj++) {
            int i = warp_offset_m + WARP_DIM_M * mi;
            int j = warp_offset_n + WARP_DIM_N * nj;
            Cpp[i * N + j] = alpha * C_rf[mi * N_local + nj] + beta * Cpp[i * N + j];
        }
    }
}

void dgemm_04_warp_tiling(DGEMM_FUNC_SIGNITURE) {
    assert(M % TILE_M == 0);
    assert(N % TILE_N == 0);
    assert(K % TILE_K == 0);
    
    dim3 gridDim(CEIL_DIV(M, TILE_M), CEIL_DIV(N, TILE_N));
    dim3 blockDim(BLOCK_DIM_X);
    kernel_dgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}