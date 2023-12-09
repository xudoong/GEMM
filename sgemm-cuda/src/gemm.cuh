#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define GEMM_FUNC_SIGNITURE int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C
#define GEMM_FUNC_PARAM M, N, K, alpha, A, B, beta, C

#define IDEAL_FAKE_KERNEL_NUMBER 2

void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE);
void gemm_01_naive(GEMM_FUNC_SIGNITURE);
void gemm_02_ideal(GEMM_FUNC_SIGNITURE);
void gemm_03_shmem(GEMM_FUNC_SIGNITURE);
void gemm_04_warp_tiling(GEMM_FUNC_SIGNITURE);
void gemm_05_shmem_plus_vectorize(GEMM_FUNC_SIGNITURE);
void gemm_06_cutlass(GEMM_FUNC_SIGNITURE);