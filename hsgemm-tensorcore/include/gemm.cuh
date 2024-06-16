#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define GEMM_FUNC_SIGNITURE int M, int N, int K, float alpha, const half *A, const half *B, float beta, float *C
#define GEMM_FUNC_PARAM M, N, K, alpha, A, B, beta, C

#define FAKE_KERNEL_NUMBER 1

void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE);
void gemm_01_fake(GEMM_FUNC_SIGNITURE);
void gemm_02_naive(GEMM_FUNC_SIGNITURE);
void gemm_03_wmma_shmem(GEMM_FUNC_SIGNITURE);
void gemm_04_wmma_shmem_opt(GEMM_FUNC_SIGNITURE);
void gemm_05_wmma_pipeline(GEMM_FUNC_SIGNITURE);