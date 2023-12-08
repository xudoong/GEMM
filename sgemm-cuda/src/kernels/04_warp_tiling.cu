#include "gemm.cuh"

const int BM = 128;
const int BN = 128;
const int BK = 32;

const int BLOCK_DIM = 256;

const int WARP_SIZE = 32;
const int WARP_DIM_M = 8;
const int WARP_DIM_N = 4;
const int NUM_WARPS_M = 2;
const int NUM_WARPS_N = 4;

const int NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;

static_assert (BLOCK_DIM == WARP_SIZE * NUM_WARPS);
static_assert (WARP_SIZE == WARP_DIM_M * WARP_DIM_N);

const int WM = BM / NUM_WARPS_M; // 64
const int WN = BN / NUM_WARPS_N; // 32

const int TM = WM / WARP_DIM_M;
const int TN = WN / WARP_DIM_N;

const int SUB_TM = 4;
const int SUB_TN = 4;

const int SUB_WM = SUB_TM * WARP_DIM_M;
const int SUB_WN = SUB_TN * WARP_DIM_N;

const int T_ITER_M = TM / SUB_TM;
const int T_ITER_N = TN / SUB_TN;


__global__ static void kernel_gemm(GEMM_FUNC_SIGNITURE) {
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BN * BK];

    float C_rf[TM * TN] = {0};
    float A_rf[TM] = {0};
    float B_rf[TN] = {0};

    const int tid = threadIdx.x;
    const int nthreads = BLOCK_DIM;

    // params for loading from global memory to shared memory
    const int stride_A_m = nthreads / BK;
    const int offset_A_m = tid / BK;
    const int offset_A_k = tid % BK;
    const int stride_B_k = nthreads / BN;
    const int offset_B_k = tid / BN;
    const int offset_B_n = tid % BN;

    // warping tiling params
    const int warp_id = tid / WARP_SIZE;
    const int warp_offset = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARPS_N;
    const int warp_id_n = warp_id % NUM_WARPS_N;
    const int warp_offset_m = warp_offset / WARP_DIM_N;
    const int warp_offset_n = warp_offset % WARP_DIM_N;

    const int Ai = blockIdx.x * BM;
    const int Bj = blockIdx.y * BN;
    A += Ai * K;
    B += Bj;
    C += Ai * N + Bj;
    
    const int Aii = warp_id_m * WM;
    const int Bjj = warp_id_n * WN;
    C += Aii * N + Bjj;
    const float *A_shared_warp = A_shared + Aii * BK;
    const float *B_shared_warp = B_shared + Bjj;

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = 0; i < BM; i += stride_A_m) {
            A_shared[(i + offset_A_m) * BK + offset_A_k] = A[(i + offset_A_m) * K + offset_A_k];
        }
        for (int i = 0; i < BK; i += stride_B_k) {
            B_shared[(i + offset_B_k) * BN + offset_B_n] = B[(i + offset_B_k) * N + offset_B_n];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // compute
        for (int k = 0; k < BK; k++) {
            for (int iter = 0; iter < T_ITER_M; iter++)
                for (int i = 0; i < SUB_TM; i++) {
                    int ii = iter * SUB_WM + warp_offset_m * SUB_TM + i;
                    A_rf[iter * SUB_TM + i] = A_shared_warp[ii * BK + k];
                }
            for (int iter = 0; iter < T_ITER_N; iter++)
                for (int j = 0; j < SUB_TN; j++) {
                    int jj = iter * SUB_WN + warp_offset_n * SUB_TN + j;
                    B_rf[iter * SUB_TN + j] = B_shared_warp[jj + k * BN];
                }
            for (int mi = 0; mi < TM; mi++) {
                for (int nj = 0; nj < TN; nj++) {
                    C_rf[mi * TN + nj] += A_rf[mi] * B_rf[nj];
                }
            }
        }
        __syncthreads();
    }

    // write to C
    for (int iterm = 0; iterm < T_ITER_M; iterm++)
    for (int itern = 0; itern < T_ITER_N; itern++)
        for (int i = 0; i < SUB_TM; i++) {
            for (int j = 0; j < SUB_TN; j++) {
                int ii = iterm * SUB_WM + warp_offset_m * SUB_TM + i;
                int jj = itern * SUB_WN + warp_offset_n * SUB_TN + j;
                C[ii * N + jj] = alpha * C_rf[(iterm * SUB_TM + i) * TN + (itern * SUB_TN + j)] + beta * C[ii * N + jj];
            }
        }
}

void gemm_04_warp_tiling(GEMM_FUNC_SIGNITURE) {
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);
    
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BLOCK_DIM);
    kernel_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}