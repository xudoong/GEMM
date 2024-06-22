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

void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE);
void gemm_01_fake(GEMM_FUNC_SIGNITURE);
void gemm_02_naive(GEMM_FUNC_SIGNITURE);
void gemm_03_wmma_shmem(GEMM_FUNC_SIGNITURE);
void gemm_04_wmma_shmem_opt(GEMM_FUNC_SIGNITURE);
void gemm_05_wmma_stage(GEMM_FUNC_SIGNITURE);
void gemm_06_wmma_stage_dbreg(GEMM_FUNC_SIGNITURE);
void gemm_07_mma_padding(GEMM_FUNC_SIGNITURE);
void gemm_08_mma_permute(GEMM_FUNC_SIGNITURE);

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

template <int xn>
__device__ static void ldmatrix(const void *p, uint32_t *reg) {
    static_assert(xn == 2 || xn == 4);
    uint32_t smem_ptr = __cvta_generic_to_shared(p);
    if (xn == 2) {
        asm volatile ( "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                       "{%0, %1}, [%2]; "
                : "=r"(reg[0]), "=r"(reg[1])
                : "r"(smem_ptr)
                );
    }
    if (xn == 4) {
        asm volatile ( "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                : "r"(smem_ptr)
                );
    }
}

__device__ static void ldmatrixA(const void *p, uint32_t *reg) {
    ldmatrix<4>(p, reg);
}

__device__ static void ldmatrixB(const void *p, uint32_t *reg) {
    ldmatrix<2>(p, reg);
}

// stmatrix requires sm_90 or higher
//__device__ static void stmatrixC(const void *p, uint32_t *reg) {
//    uint32_t smem_ptr = __cvta_generic_to_shared(p);
//    asm volatile ( "stmatrix.sync.aligned.x2.m8n8.shared.b16 "
//                   "[%0], {%1, %2}; "
//            : "=r"(smem_ptr)
//            : "r"(reg[0]), "r"(reg[1])
//            );
//}

__device__ static void mma_m16n8k16(uint32_t *rA, uint32_t *rB, uint32_t *rC) {
    asm( "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
         " { %0, %1 }, "
         " { %2, %3, %4, %5 }, "
         " { %6, %7 }, "
         " { %8, %9};"
            : "=r"(rC[0]), "=r"(rC[1])
            : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
    "r"(rB[0]), "r"(rB[1]),
    "r"(rC[0]), "r"(rC[1])
            );
}
