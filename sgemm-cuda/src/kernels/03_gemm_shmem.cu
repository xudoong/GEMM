#include "gemm.cuh"

const int BM = 128;
const int BN = 128;
const int BK = 32;

const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

const int TM = 8;
const int TN = 8;


__global__ static void kernel_gemm(GEMM_FUNC_SIGNITURE) {
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BN * BK];
  
    int Ai = blockIdx.x * BM;
    int Bj = blockIdx.y * BN;

    A += Ai * K;
    B += Bj;
    C += Ai * N + Bj;

    const int M_local = BM / BLOCK_DIM_Y;
    const int N_local = BN / BLOCK_DIM_X;

    assert (M_local == TM);
    assert (N_local == TN);
    
    float C_rf[M_local * N_local] = {0};
    float A_rf[M_local] = {0};
    float B_rf[N_local] = {0};

    // version 1 fast
    const int tid = threadIdx.x;
    // version 2 slow
    // const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    const int tid_x = tid % BLOCK_DIM_X;
    const int tid_y = tid / BLOCK_DIM_X;
    const int nthreads = BLOCK_DIM_X * BLOCK_DIM_Y;
    const int stride_A_m = nthreads / BK;
    const int offset_A_m = tid / BK;
    const int offset_A_k = tid % BK;
    const int stride_B_k = nthreads / BN;
    const int offset_B_k = tid / BN;
    const int offset_B_n = tid % BN;

    for (int kb = 0; kb < K; kb += BK) {
        // version 1 fast
        for (int i = 0; i < BM; i += stride_A_m) {
            A_shared[(i + offset_A_m) * BK + offset_A_k] = A[(i + offset_A_m) * K + offset_A_k];
        }
        for (int i = 0; i < BK; i += stride_B_k) {
            B_shared[(i + offset_B_k) * BN + offset_B_n] = B[(i + offset_B_k) * N + offset_B_n];
        }

        // version 2 slow
        // for (int i = offset_A_m; i < BM; i += stride_A_m) {
        //     A_shared[(i) * BK + offset_A_k] = A[(i) * K + offset_A_k];
        // }
        // for (int i = offset_B_k; i < BK; i += stride_B_k) {
        //   B_shared[(i) * BN + offset_B_n] = B[(i) * N + offset_B_n];
        // }
        __syncthreads();

        A += BK;
        B += BK * N;

        // compute
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                int ii = tid_y * TM + i;
                A_rf[i] = A_shared[ii * BK + k];
            }
            for (int j = 0; j < TN; j++) {
                int jj = tid_x * TN + j;
                B_rf[j] = B_shared[jj + k * BN];
            }
            for (int mi = 0; mi < M_local; mi++) {
                for (int nj = 0; nj < N_local; nj++) {
                    C_rf[mi * N_local + nj] += A_rf[mi] * B_rf[nj];
                }
            }
        }
        __syncthreads();
    }

    // write to C
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int ii = tid_y * TM + i;
            int jj = tid_x * TN + j;
            C[ii * N + jj] = alpha * C_rf[i * N_local + j] + beta * C[ii * N + jj];
        }
    }
}

void gemm_03_shmem(GEMM_FUNC_SIGNITURE) {
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);
    
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BLOCK_DIM_X * BLOCK_DIM_Y);
    kernel_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}