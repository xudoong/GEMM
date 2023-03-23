#include "dgemm.h"

#define BK 16
#define BM 16
#define BN 16


void kernel8x8x8(const double *A, int lda, const double *B, int ldb, double *C, int ldc, double alpha) 
{
    __m512d b[8], c[8], v;
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + ldb * i);
        c[i] = _mm512_load_pd(C + ldc * i);
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            v = _mm512_set1_pd(alpha * A(i, j));
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]);
            _mm512_store_pd(C + ldc * i, c[i]);
        }
    }
}


// C = alpha * A * B + beta * C
void dgemm_vec(DGEMM_FUNC_SIGNITURE)
{

    if (beta != 1)
        scale(C, M * N, beta);
    
    for (int bk = 0; bk < K; bk += BK)
        for (int bm = 0; bm < M; bm += BM)
            for (int bn = 0; bn < N; bn += BN)
            {
                for (int k = bk; k < MIN(bk+BK, K); k+=8) {
                    for (int i = bm; i < MIN(bm+BM, M); i+=8) {
                        for (int j = bn; j < MIN(bn+BN, N); j+=8) {
                            const double *Ap = A + i * lda + k;
                            const double *Bp = B + k * ldb + j;
                            double *Cp = C + i * ldc + j;
                            kernel8x8x8(Ap, lda, Bp, ldb, Cp, ldc, alpha);
                        }
                    }
                }
            }
}

