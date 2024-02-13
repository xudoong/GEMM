#include "gemm.cuh"

const int BM = 128;
const int BN = 128;
const int BK = 32;

const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

const int TM = 8;
const int TN = 8;

#define AS_FLOAT4(ptr) (reinterpret_cast<float4 *>(ptr))
#define AS_CONST_FLOAT4(ptr) (reinterpret_cast<const float4 *>(ptr))

__global__ static void kernel_gemm(GEMM_FUNC_SIGNITURE) {
    __shared__ float A_shared[BM * (BK + 4)];
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

    const int tid = threadIdx.x;

    const int tid_x = tid % BLOCK_DIM_X;
    const int tid_y = tid / BLOCK_DIM_X;
    const int nthreads = BLOCK_DIM_X * BLOCK_DIM_Y;
    const int stride_A_m = nthreads / (BK / 4);
    const int offset_A_m = tid / (BK / 4);
    const int offset_A_k = tid % (BK / 4) * 4;
    const int stride_B_k = nthreads / (BN / 4);
    const int offset_B_k = tid / (BN / 4);
    const int offset_B_n = tid % (BN / 4) * 4;

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = 0; i < BM; i += stride_A_m) {
            AS_FLOAT4(&A_shared[(i + offset_A_m) * (BK + 4) + offset_A_k])[0] = AS_CONST_FLOAT4(&A[(i + offset_A_m) * K + offset_A_k])[0];
            
        }
        for (int i = 0; i < BK; i += stride_B_k) {
            AS_FLOAT4(&B_shared[(i + offset_B_k) * BN + offset_B_n])[0] = AS_CONST_FLOAT4(&B[(i + offset_B_k) * N + offset_B_n])[0];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // compute
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                int ii = tid_y * TM + i;
                A_rf[i] = A_shared[ii * (BK + 4) + k];
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
        for (int j = 0; j < TN; j+=4) {
            int ii = tid_y * TM + i;
            int jj = tid_x * TN + j;
            float4 tmp = AS_CONST_FLOAT4(&C[ii * N + jj])[0];
            tmp.x = alpha * C_rf[i * N_local + j] + beta * tmp.x;
            tmp.y = alpha * C_rf[i * N_local + j + 1] + beta * tmp.y;
            tmp.z = alpha * C_rf[i * N_local + j + 2] + beta * tmp.z;
            tmp.w = alpha * C_rf[i * N_local + j + 3] + beta * tmp.w;

            AS_FLOAT4(&C[ii * N + jj])[0] = tmp;
        }
    }
}

void gemm_07_rm_bank_conflict(GEMM_FUNC_SIGNITURE) {
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);
    
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BLOCK_DIM_X * BLOCK_DIM_Y);
    kernel_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}