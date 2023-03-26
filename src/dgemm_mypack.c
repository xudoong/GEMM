#include "dgemm.h"

#define BK 1024
#define BM 1024
#define BN 1024
#define INNER_BLK_SIZE_M 16
#define INNER_BLK_SIZE_N 8
#define INNER_BLK_SIZE_K 64
#define MYPACK_KERNEL kernel16x8x64

static double *Abuf = NULL;
static double *Bbuf = NULL;
static double *Cbuf = NULL;

static int Abuf_len = 4096 * 4096;
static int Bbuf_len = 4096 * 4096;
static int Cbuf_len = 4096 * 4096;

/* kernel 16x8xk */
#define kernel16x8xk_block_start \
    __m512d b[16], c[16]; \
    for (int i = 0; i < 16; i++) { \
        b[i] = _mm512_load_pd(B + i * 8); \
    } \
    for (int i = 0; i < 16; i++) { \
        c[i] = _mm512_load_pd(C + i * 8); \
    }
#define kernel16x8xk_block_end \
    for (int i = 0; i < 16; i++) { \
        _mm512_store_pd(C + i * 8, c[i]); \
    }

#define kernel16x8xk_block(idx) \
    for (int j = 0; j < 16; j++) { \
        for (int i = 0; i < 16; i++) { \
            __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 16) * 16]); \
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
        } \
        b[j] = _mm512_load_pd(B + (j + (idx + 1) * 16) * 8); \
    }
#define kernel16x8xk_block_last(idx) \
    for (int j = 0; j < 16; j++) { \
        for (int i = 0; i < 16; i++) { \
            __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 16) * 16]); \
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
        } \
    }

/* kernel 24x8xk */

#define kernel24x8xk_block_start \
    __m512d b[8], c[24]; \
    for (int i = 0; i < 8; i++) { \
        b[i] = _mm512_load_pd(B + i * 8); \
    } \
    for (int i = 0; i < 24; i++) { \
        c[i] = _mm512_load_pd(C + i * 8); \
    }
#define kernel24x8xk_block_end \
    for (int i = 0; i < 24; i++) { \
        _mm512_store_pd(C + i * 8, c[i]); \
    }

#define kernel24x8xk_block(idx) \
    for (int j = 0; j < 8; j++) { \
        _Pragma("GCC unroll 24") \
        for (int i = 0; i < 24; i++) { \
            __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 8) * 24]); \
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
        } \
        b[j] = _mm512_load_pd(B + (j + (idx + 1) * 8) * 8); \
    }
#define kernel24x8xk_block_last(idx) \
    for (int j = 0; j < 8; j++) { \
        _Pragma("GCC unroll 24") \
        for (int i = 0; i < 24; i++) { \
            __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 8) * 24]); \
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
        } \
    }

// /* kernel 16x8xk with prefetch */
// #define kernel16x8xk_block_start \
//     __m512d b[16], c[16]; \
//     _mm_prefetch(A, _MM_HINT_T0); \
//     for (int i = 0; i < 16; i++) { \
//         b[i] = _mm512_load_pd(B + i * 8); \
//     } \
//     for (int i = 0; i < 16; i++) { \
//         c[i] = _mm512_load_pd(C + i * 8); \
//     }
// #define kernel16x8xk_block_end \
//     for (int i = 0; i < 16; i++) { \
//         _mm512_store_pd(C + i * 8, c[i]); \
//         _mm_prefetch(C + (i + 16) * 8, _MM_HINT_T0); \
//     }

// #define kernel16x8xk_block(idx) \
//     for (int j = 0; j < 16; j++) { \
//         for (int i = 0; i < 16; i++) { \
//             __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 16) * 16]); \
//             c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
//         } \
//         b[j] = _mm512_load_pd(B + (j + (idx + 1) * 16) * 8); \
//     }
// #define kernel16x8xk_block_last(idx) \
//     for (int j = 0; j < 16; j++) { \
//         for (int i = 0; i < 16; i++) { \
//             __m512d v = _mm512_set1_pd(A[i + (j + (idx) * 16) * 16]); \
//             c[i] = _mm512_fmadd_pd(v, b[j], c[i]); \
//         } \
//         _mm_prefetch(B + (j + (idx + 1) * 16) * 8, _MM_HINT_T0); \
//     }
    
static void kernel8x8x8(const double *A, const double *B, double *C)
{
    __m512d b[8], c[8], v[8];
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
        c[i] = _mm512_load_pd(C + i * 8);
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            v[i] = _mm512_set1_pd(A[i + j * 8]);
            c[i] = _mm512_fmadd_pd(v[i], b[j], c[i]);
            _mm512_store_pd(C + i * 8, c[i]);
        } 
    }
}

static void kernel16x8x8(const double *A, const double *B, double *C)
{
    __m512d b[8], c[16], v[8];
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
    }
    for (int i = 0; i < 16; i++) {
        c[i] = _mm512_load_pd(C + i * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 16; i++) {
            v[i % 8] = _mm512_set1_pd(A[i + j * 16]);
            c[i] = _mm512_fmadd_pd(v[i % 8], b[j], c[i]);
        } 
    }
    for (int i = 0; i < 8; i++) {
        _mm512_store_pd(C + i * 8, c[i]);
    }

}

static void kernel16x8x16(const double *A, const double *B, double *C)
{
    __m512d b[16], c[16];
    for (int i = 0; i < 16; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
    }
    for (int i = 0; i < 16; i++) {
        c[i] = _mm512_load_pd(C + i * 8);
    }
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            __m512d v = _mm512_set1_pd(A[i + j * 16]);
            c[i] = _mm512_fmadd_pd(v, b[j], c[i]);
        } 
    }
    for (int i = 0; i < 16; i++) {
        _mm512_store_pd(C + i * 8, c[i]);
    }
}

static void kernel16x8x32(const double *A, const double *B, double *C)
{
    kernel16x8xk_block_start
    kernel16x8xk_block(0)
    kernel16x8xk_block(1)
    kernel16x8xk_block_end
}

static void kernel16x8x64(const double *A, const double *B, double *C)
{
    kernel16x8xk_block_start
    kernel16x8xk_block(0)
    kernel16x8xk_block(1)
    kernel16x8xk_block(2)
    kernel16x8xk_block_last(3)
    kernel16x8xk_block_end
}

static void kernel24x8x16(const double *A, const double *B, double *C)
{
    kernel24x8xk_block_start
    kernel24x8xk_block(0)
    kernel24x8xk_block_last(1)
    kernel24x8xk_block_end
}

static void kernel24x8x32(const double *A, const double *B, double *C)
{
    kernel24x8xk_block_start
    kernel24x8xk_block(0)
    kernel24x8xk_block(1)
    kernel24x8xk_block(2)
    kernel24x8xk_block_last(3)
    kernel24x8xk_block_end
}

static void kernel24x8x64(const double *A, const double *B, double *C)
{
    kernel24x8xk_block_start
    kernel24x8xk_block(0)
    kernel24x8xk_block(1)
    kernel24x8xk_block(2)
    kernel24x8xk_block(3)
    kernel24x8xk_block(4)
    kernel24x8xk_block(5)
    kernel24x8xk_block(6)
    kernel24x8xk_block_last(7)
    kernel24x8xk_block_end
}


static void kernel32x8x8(const double *A, const double *B, double *C)
{
    __m512d b[8], c[16], v[8];
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
    }

    for (int i = 0; i < 16; i++) {
        c[i] = _mm512_load_pd(C + i * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 16; i++) {
            v[i % 8] = _mm512_set1_pd(A[i + j * 32]);
            c[i] = _mm512_fmadd_pd(v[i % 8], b[j], c[i]);
            _mm512_store_pd(C + i * 8, c[i]);
        } 
    }

    for (int i = 0; i < 16; i++) {
        c[i] = _mm512_load_pd(C + (i + 16) * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 16; i++) {
            v[i % 8] = _mm512_set1_pd(A[i + 16 + j * 32]);
            c[i] = _mm512_fmadd_pd(v[i % 8], b[j], c[i]);
            _mm512_store_pd(C + (i + 16) * 8, c[i]);
        } 
    }
}

static void kernel32x8x8_v2(const double *A, const double *B, double *C)
{
    __m512d b[8], c[12], v[12];
    for (int i = 0; i < 8; i++) {
        b[i] = _mm512_load_pd(B + i * 8);
    }

    for (int i = 0; i < 12; i++) {
        c[i] = _mm512_load_pd(C + i * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 12; i++) {
            v[i] = _mm512_set1_pd(A[i + j * 32]);
            c[i] = _mm512_fmadd_pd(v[i], b[j], c[i]);
            _mm512_store_pd(C + i * 8, c[i]);
        } 
    }
    for (int i = 0; i < 12; i++) {
        c[i] = _mm512_load_pd(C + (i + 12) * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 12; i++) {
            v[i] = _mm512_set1_pd(A[(i + 12) + j * 32]);
            c[i] = _mm512_fmadd_pd(v[i], b[j], c[i]);
            _mm512_store_pd(C + (i + 12) * 8, c[i]);
        } 
    }
    for (int i = 0; i < 8; i++) {
        c[i] = _mm512_load_pd(C + (i + 24) * 8);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            v[i] = _mm512_set1_pd(A[(i + 24) + j * 32]);
            c[i] = _mm512_fmadd_pd(v[i], b[j], c[i]);
            _mm512_store_pd(C + (i + 24) * 8, c[i]);
        } 
    }
}

// store inner most block of A as col major
static void do_packA(int M, int K, const double *A)
{
    for (int bk = 0; bk < K; bk += BK)
        for (int bm = 0; bm < M; bm += BM)
            for (int k = bk; k < MIN(bk + BK, K); k += INNER_BLK_SIZE_K)
                for (int i = bm; i < MIN(bm + BM, M); i += INNER_BLK_SIZE_M)
                {
                    int offset = (bk * M + bm * BK) + ((k - bk) * BM + (i - bm) * INNER_BLK_SIZE_K);
                    for (int ii = 0; ii < INNER_BLK_SIZE_M; ii++)
                        for (int kk = 0; kk < INNER_BLK_SIZE_K; kk++)
                            Abuf[offset + kk * INNER_BLK_SIZE_M + ii] = A[(i + ii) * K + k + kk];
                }
}

static void do_packB(int K, int N, const double *B)
{
    for (int bk = 0; bk < K; bk += BK)
        for (int bn = 0; bn < N; bn += BN)
            for (int k = bk; k < MIN(bk + BK, K); k += INNER_BLK_SIZE_K)
                for (int j = bn; j < MIN(bn + BN, N); j += INNER_BLK_SIZE_N)
                {
                    int offset = (bk * N + bn * BK) + ((k - bk) * BN + (j - bn) * INNER_BLK_SIZE_K);
                    for (int kk = 0; kk < INNER_BLK_SIZE_K; kk++)
                        memcpy(Bbuf + offset + kk * INNER_BLK_SIZE_N, &B[(k + kk) * N + j], sizeof(double) * INNER_BLK_SIZE_N); 
                }
}

static void do_unpackC(int M, int N, double *C, double alpha)
{       
    for (int bm = 0; bm < M; bm += BM)
        for (int bn = 0; bn < N; bn += BN)
            for (int i = bm; i < MIN(bm + BM, M); i += INNER_BLK_SIZE_M)
                for (int j = bn; j < MIN(bn + BN, N); j += INNER_BLK_SIZE_N)
                {
                    int offset = (bm * N + bn * BM) + ((i - bm) * BN + (j - bn) * INNER_BLK_SIZE_M);
                    for (int ii = 0; ii < INNER_BLK_SIZE_M; ii++)
                        for (int jj = 0; jj < INNER_BLK_SIZE_N; jj++)
                            C[(i + ii) * N + (j + jj)] += alpha * Cbuf[offset + ii * INNER_BLK_SIZE_N + jj];
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
                for (int k = bk; k < MIN(bk+BK, K); k+=INNER_BLK_SIZE_K) {
                    for (int i = bm; i < MIN(bm+BM, M); i+=INNER_BLK_SIZE_M) {
                        for (int j = bn; j < MIN(bn+BN, N); j+=INNER_BLK_SIZE_N) {
                            int Aoffset = (bk * M + bm * BK) + ((k - bk) * BM + (i - bm) * INNER_BLK_SIZE_K);
                            int Boffset = (bk * N + bn * BK) + ((k - bk) * BN + (j - bn) * INNER_BLK_SIZE_K);
                            int Coffset = (bm * N + bn * BM) + ((i - bm) * BN + (j - bn) * INNER_BLK_SIZE_M);
                            MYPACK_KERNEL(Abuf + Aoffset, Bbuf + Boffset, Cbuf+Coffset);
                        }
                    }
                }
            }

    do_unpackC(M, N, C, alpha);
}
