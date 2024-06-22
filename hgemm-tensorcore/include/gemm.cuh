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

#define GEMM_FUNC_SIGNITURE int M, int N, int K, half alpha, const half *A, const half *B, half beta, half *C
#define GEMM_FUNC_PARAM M, N, K, alpha, A, B, beta, C

#define FAKE_KERNEL_NUMBER 1

template <size_t size>
__device__ static void copy_async(void *dst, const void *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), "l"(src), "n"(size));
}

__device__ static void copy_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ static void copy_async_wait_and_syncthreads() {
    if (n >= 0) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
    }
    __syncthreads();
}

void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE);
void gemm_01_fake(GEMM_FUNC_SIGNITURE);
void gemm_02_naive(GEMM_FUNC_SIGNITURE);
void gemm_03_wmma_shmem(GEMM_FUNC_SIGNITURE);
void gemm_04_wmma_shmem_opt(GEMM_FUNC_SIGNITURE);
void gemm_05_wmma_stage(GEMM_FUNC_SIGNITURE);
void gemm_06_wmma_stage_dbreg(GEMM_FUNC_SIGNITURE);
