#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define DGEMM_FUNC_SIGNITURE int M, int N, int K, double alpha, const double *A, const double *B, double beta, double *C
#define DGEMM_FUNC_PARAM M, N, K, alpha, A, B, beta, C

#define IDEAL_FAKE_KERNEL_NUMBER 2

void dgemm_00_cublas(cublasHandle_t handle, DGEMM_FUNC_SIGNITURE);
void dgemm_01_naive(DGEMM_FUNC_SIGNITURE);
void dgemm_02_ideal(DGEMM_FUNC_SIGNITURE);