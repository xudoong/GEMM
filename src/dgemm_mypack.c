#include "dgemm.h"

#define BK 64
#define BM 64
#define BN 64
#define INNER_BLK_SIZE 8
#define INNER_BLK_SIZE_SQUARE 64

#define BLK_IDX(i, block_size) ((i) % (block_size))

static double *Abuf = NULL;
static double *Bbuf = NULL;
static double *Cbuf = NULL;

static int Abuf_len = 4096 * 4096;
static int Bbuf_len = 4096 * 4096;
static int Cbuf_len = 4096 * 4096;

static void kernel8x8x8(const double *A, const double *B, double *C)
{
    __m512d b[8], c[8], v[8];
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
        c[i] = _mm512_load_pd(C + i * 8);
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i+=4) {
            v[(i + 0)] = _mm512_set1_pd(A[(i + 0) * 8 + j]);
            v[(i + 1)] = _mm512_set1_pd(A[(i + 1) * 8 + j]);
            v[(i + 2)] = _mm512_set1_pd(A[(i + 2) * 8 + j]);
            v[(i + 3)] = _mm512_set1_pd(A[(i + 3) * 8 + j]);
            c[(i + 0)] = _mm512_fmadd_pd(v[i + 0], b[j], c[(i + 0)]);
            c[(i + 1)] = _mm512_fmadd_pd(v[i + 1], b[j], c[(i + 1)]);
            c[(i + 2)] = _mm512_fmadd_pd(v[i + 2], b[j], c[(i + 2)]);
            c[(i + 3)] = _mm512_fmadd_pd(v[i + 3], b[j], c[(i + 3)]);
            _mm512_store_pd(C + (i + 0) * 8, c[i + 0]);
            _mm512_store_pd(C + (i + 1) * 8, c[i + 1]);
            _mm512_store_pd(C + (i + 2) * 8, c[i + 2]);
            _mm512_store_pd(C + (i + 3) * 8, c[i + 3]);
        } 
    }
}

static void do_packA(int M, int K, const double *A)
{
    for (int bk = 0; bk < K; bk += BK)
        for (int bm = 0; bm < M; bm += BM)
            for (int k = bk; k < MIN(bk + BK, K); k += INNER_BLK_SIZE)
                for (int i = bm; i < MIN(bm + BM, M); i += INNER_BLK_SIZE)
                {
                    int offset = (bk * M + bm * BK) + ((k - bk) * BM + (i - bm) * INNER_BLK_SIZE);
                    for (int ii = 0; ii < INNER_BLK_SIZE; ii++)
                        memcpy(Abuf + offset + ii * INNER_BLK_SIZE, &A[(i + ii) * K + k], sizeof(double) * INNER_BLK_SIZE);
                }
}

static void do_packB(int K, int N, const double *B)
{
    for (int bk = 0; bk < K; bk += BK)
        for (int bn = 0; bn < N; bn += BN)
            for (int k = bk; k < MIN(bk + BK, K); k += INNER_BLK_SIZE)
                for (int j = bn; j < MIN(bn + BN, N); j += INNER_BLK_SIZE)
                {
                    int offset = (bk * N + bn * BK) + ((k - bk) * BN + (j - bn) * INNER_BLK_SIZE);
                    for (int kk = 0; kk < INNER_BLK_SIZE; kk++)
                        memcpy(Bbuf + offset + kk * INNER_BLK_SIZE, &B[(k + kk) * N + j], sizeof(double) * INNER_BLK_SIZE); 
                }
}

static void do_unpackC(int M, int N, double *C, double alpha)
{       
    for (int bm = 0; bm < M; bm += BM)
        for (int bn = 0; bn < N; bn += BN)
            for (int i = bm; i < MIN(bm + BM, M); i += INNER_BLK_SIZE)
                for (int j = bn; j < MIN(bn + BN, N); j += INNER_BLK_SIZE)
                {
                    int offset = (bm * N + bn * BM) + ((i - bm) * BN + (j - bn) * INNER_BLK_SIZE);
                    for (int ii = 0; ii < INNER_BLK_SIZE; ii++)
                        for (int jj = 0; jj < INNER_BLK_SIZE; jj++)
                            C[(i + ii) * N + (j + jj)] += alpha * Cbuf[offset + ii * INNER_BLK_SIZE + jj];
                }
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

// dgemm_pack is defined in MKL :)
void dgemm_mypack(DGEMM_FUNC_SIGNITURE)
{

    if (beta != 1)
        scale(C, M * N, beta);
    
    my_pack(M, N, K, A, B);

    for (int bk = 0; bk < K; bk += BK)
        for (int bm = 0; bm < M; bm += BM)
            for (int bn = 0; bn < N; bn += BN)
            {
                for (int k = bk; k < MIN(bk+BK, K); k+=8) {
                    for (int i = bm; i < MIN(bm+BM, M); i+=8) {
                        for (int j = bn; j < MIN(bn+BN, N); j+=8) {
                            int Aoffset = (bk * M + bm * BK) + ((k - bk) * BM + (i - bm) * INNER_BLK_SIZE);
                            int Boffset = (bk * N + bn * BK) + ((k - bk) * BN + (j - bn) * INNER_BLK_SIZE);
                            int Coffset = (bm * N + bn * BM) + ((i - bm) * BN + (j - bn) * INNER_BLK_SIZE);
                            kernel8x8x8(Abuf + Aoffset, Bbuf + Boffset, Cbuf+Coffset);
                        }
                    }
                }
            }

    do_unpackC(M, N, C, alpha);
}
