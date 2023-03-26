#include "dgemm.h"

/* Strategy similar to OpenBLAS */

#define BM 8
#define BN 24

static double *Abuf = NULL;
static double *Bbuf = NULL;
static double *Cbuf = NULL;

static int Abuf_len = 4096 * 4096;
static int Bbuf_len = 4096 * 4096;
static int Cbuf_len = 4096 * 4096;

static void kernel_n24m8(const double * __restrict__ A, const double * __restrict__ B, double * __restrict__ C, const int K)
{
    __m512d b[3], c[24];

    // load
    _Pragma("GCC unroll 24")
    for (int i = 0; i < 24; i++)
        c[i] = _mm512_load_pd(C + i * 8);

    // compute
    for (int k = 0; k < K; k++) {
        b[0] = _mm512_load_pd(B + k * 24 + 0);
        b[1] = _mm512_load_pd(B + k * 24 + 8);
        b[2] = _mm512_load_pd(B + k * 24 + 16);

        
        for (int i = 0; i < 8; i++) {
            __m512d a = _mm512_set1_pd(A[k * 8 + i]);
            c[i * 3 + 0] = _mm512_fmadd_pd(a, b[0], c[i * 3 + 0]);
            c[i * 3 + 1] = _mm512_fmadd_pd(a, b[1], c[i * 3 + 1]);
            c[i * 3 + 2] = _mm512_fmadd_pd(a, b[2], c[i * 3 + 2]);

        }
    }

    // store
    _Pragma("GCC unroll 24")
    for (int i = 0; i < 24; i++)
        _mm512_store_pd(C + i * 8, c[i]);
}


static void do_packA(int M, int K, const double *A)
{
    for (int bm = 0; bm < M; bm += BM)
        for (int k = 0; k < K; k++)
            for (int i = 0; i < BM; i++)
                Abuf[bm * K + k * BM + i] = A[(bm + i) * K + k];
}

static void do_packB(int K, int N, const double *B)
{
    for (int bn = 0; bn < N; bn += BN)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < BN; j++)
                Bbuf[bn * K + k * BN + j] = B[k * N + bn + j];
}

static void do_unpackC(int M, int N, double *C, double alpha)
{
    for (int bn = 0; bn < N; bn += BN)
        for (int bm = 0; bm < M; bm += BM)
            for (int i = 0; i < BM; i++)
                for (int j = 0; j < BN; j++)
                    C[(bm + i) * N + bn + j] += alpha * Cbuf[bn * M + bm * BN + i * BN + j];

}

static void my_pack(int M, int N, int K, const double *A, const double *B)
{
    if (M * K > Abuf_len || K * N > Bbuf_len) {
        printf("Error: packing buffer size is not enough.\n");
        exit(-1);
    }
    if (Abuf == NULL) {
        Abuf = (double *) _mm_malloc(Abuf_len * sizeof(double), 64);
    }
    if (Bbuf == NULL) {
        Bbuf = (double *) _mm_malloc(Bbuf_len * sizeof(double), 64);
    }
    if (Cbuf == NULL) {
        Cbuf = (double *) _mm_malloc(Cbuf_len * sizeof(double), 64);
    }

    do_packA(M, K, A);
    do_packB(K, N, B);
    memset(Cbuf, 0, sizeof(double) * M * N);
}

void dgemm_v1(DGEMM_FUNC_SIGNITURE)
{

    if (beta != 1)
        scale(C, M * N, beta);
    
    my_pack(M, N, K, A, B);

    for (int bn = 0; bn < N; bn += BN)
    {
        for (int bm = 0; bm < M; bm += BM)
        {
            kernel_n24m8(Abuf + bm * K, Bbuf + bn * K, Cbuf + bn * M + bm * BN, K);
        }
    }

    do_unpackC(M, N, C, alpha);

}
