#ifndef __XD_DGEMM_H__
#define __XD_DGEMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#define A(i, j) A[(i) * lda + (j)]
#define B(i, j) B[(i) * ldb + (j)]
#define C(i, j) C[(i) * ldc + (j)]

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define CEIL_DIV(x, d) (((x) + (d) - 1) / (d))
#define CEIL(x, d) (CEIL_DIV(x, d) * d)

#define DGEMM_FUNC_SIGNITURE int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc

void dgemm_mkl(DGEMM_FUNC_SIGNITURE);
void dgemm_openblas(DGEMM_FUNC_SIGNITURE);
void dgemm_naive(DGEMM_FUNC_SIGNITURE);
void dgemm_ideal(DGEMM_FUNC_SIGNITURE);
void dgemm_vec(DGEMM_FUNC_SIGNITURE);
void dgemm_mypack(DGEMM_FUNC_SIGNITURE);
void dgemm_v1(DGEMM_FUNC_SIGNITURE);
void dgemm_v2(DGEMM_FUNC_SIGNITURE);
void dgemm_v3(DGEMM_FUNC_SIGNITURE);
void dgemm_v4(DGEMM_FUNC_SIGNITURE);

void scale(double *v, int len, double scale);

#endif