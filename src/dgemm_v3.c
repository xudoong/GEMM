#include "dgemm.h"

/* V1 + Optimize packing + TILING on M & K */

#define BM 8
#define BN 24

#define TILE_M 1024
#define TILE_N 1920
#define TILE_K 1024

static double *Abuf = NULL;
static double *Bbuf = NULL;

static int Abuf_len = 4096 * 4096;
static int Bbuf_len = 4096 * 4096;

static void kernel_n24m8(const double * __restrict__ A, const double * __restrict__ B, double * __restrict__ C, const int ldc, const int Kt, double alpha, double beta)
{
    __m512d c[24];

    // zero vector c
    _Pragma("GCC unroll 24")
    for (int i = 0; i < 24; i++)
        c[i] = _mm512_xor_pd(c[i], c[i]);

    // prefetch C to L2
    for (int i = 0; i < 8; i++) {
        _mm_prefetch(C + i * ldc + 0, _MM_HINT_T1);
    }

    // compute
    for (int k = 0; k < Kt; k++) {
        __m512d b[3];
        b[0] = _mm512_load_pd(B + k * 24 + 0);
        b[1] = _mm512_load_pd(B + k * 24 + 8);
        b[2] = _mm512_load_pd(B + k * 24 + 16);

        _mm_prefetch(B + (k + 4) * 24 + 0, _MM_HINT_T0);
        _mm_prefetch(B + (k + 4) * 24 + 8, _MM_HINT_T0);
        _mm_prefetch(B + (k + 4) * 24 + 16, _MM_HINT_T0);
        _mm_prefetch(A + (k + 4) * 8, _MM_HINT_T0);
        
        for (int i = 0; i < 8; i++) {
            __m512d a = _mm512_set1_pd(A[k * 8 + i]);
            c[i * 3 + 0] = _mm512_fmadd_pd(a, b[0], c[i * 3 + 0]);
            c[i * 3 + 1] = _mm512_fmadd_pd(a, b[1], c[i * 3 + 1]);
            c[i * 3 + 2] = _mm512_fmadd_pd(a, b[2], c[i * 3 + 2]);
        }
    }

    // store
    __m512d v_alpha = _mm512_set1_pd(alpha);
    __m512d v_beta = _mm512_set1_pd(beta);

    _Pragma("GCC unroll 8")
    for (int i = 0; i < 8; i++) {
        __m512d out[3];
        out[0] = _mm512_load_pd(C + i * ldc + 0);
        out[1] = _mm512_load_pd(C + i * ldc + 8);
        out[2] = _mm512_load_pd(C + i * ldc + 16);

        if (i < 4) {
            _mm_prefetch(C + (i + 4) * ldc + 0, _MM_HINT_T0);
            _mm_prefetch(C + (i + 4) * ldc + 8, _MM_HINT_T0);
            _mm_prefetch(C + (i + 4) * ldc + 16, _MM_HINT_T0);
        }

        out[0] = _mm512_mul_pd(out[0], v_beta);
        out[1] = _mm512_mul_pd(out[1], v_beta);
        out[2] = _mm512_mul_pd(out[2], v_beta);

        out[0] = _mm512_fmadd_pd(v_alpha, c[i * 3 + 0], out[0]);
        out[1] = _mm512_fmadd_pd(v_alpha, c[i * 3 + 1], out[1]);
        out[2] = _mm512_fmadd_pd(v_alpha, c[i * 3 + 2], out[2]);

        _mm512_stream_pd(C + i * ldc + 0,  out[0]);
        _mm512_stream_pd(C + i * ldc + 8,  out[1]);
        _mm512_stream_pd(C + i * ldc + 16, out[2]);
    }
}

static void do_packA_bm(int Abuf_offset, int bm, int Kt, int Mb, int lda, const double *Ap)
{
    for (int k = 0; k < Kt; k++)
        for (int i = 0; i < Mb; i++)
            Abuf[Abuf_offset + k * Mb + i] = Ap[(bm + i) * lda + k];
}

static void do_packB_bn(int Bbuf_offset, int bn, int Kt, int Nb, int ldb, const double *Bp)
{
    for (int k = 0; k < Kt; k++)
        for (int j = 0; j < Nb; j++)
            Bbuf[Bbuf_offset + k * Nb + j] = Bp[k * ldb + bn + j];
}

static void init_pack_buffer(int M, int N, int K)
{
    if (M * K > Abuf_len || K * N > Bbuf_len) {
        printf("Error: the packing buffer size is not enough.\n");
        exit(-1);
    }

    if (Abuf == NULL) {
        Abuf = (double *) _mm_malloc(Abuf_len * sizeof(double), 64);
    }
    if (Bbuf == NULL) {
        Bbuf = (double *) _mm_malloc(Bbuf_len * sizeof(double), 64);
    }
}

void dgemm_v3(DGEMM_FUNC_SIGNITURE)
{
    if (M % BM != 0 || N % BN != 0) {
        printf("Error: M, N must be a multiple of (%d, %d).\n", BM, BN);
        exit(-1);
    }

    init_pack_buffer(M, N, K);

    // tiling
    for (int tn = 0; tn < N; tn += TILE_N) {
        for (int tm = 0; tm < M; tm += TILE_M) {
            for (int tk = 0; tk < K; tk += TILE_K) {
                const double *Ap = A + tm * K + tk;
                const double *Bp = B + tk * N + tn;
                double *Cp = C + tm * N + tn;

                int Mt = MIN(TILE_M, M - tm);
                int Nt = MIN(TILE_N, N - tn);
                int Kt = MIN(TILE_K, K - tk);

                // the inner is similar to dgemm_v2
                for (int bn = 0; bn < Nt; bn += BN) {
                    int Nb = MIN(BN, Nt - bn);
                    int Bbuf_offset = tn * K + tk * Nt + bn * Kt;
                    if (tm == 0) {
                        do_packB_bn(Bbuf_offset, bn, Kt, Nb, ldb, Bp);
                    }

                    for (int bm = 0; bm < Mt; bm += BM) {
                        int Mb = MIN(BM, Mt - bm);
                        int Abuf_offset = tm * K + tk * Mt + bm * Kt;

                        if (tn == 0 && bn == 0) {
                            do_packA_bm(Abuf_offset, bm, Kt, Mb, lda, Ap);
                        }

                        double beta_or_1 = (tk == 0) ? beta : 1;

                        kernel_n24m8(Abuf + Abuf_offset, Bbuf + Bbuf_offset, Cp + bm * ldc + bn, ldc, Kt, alpha, beta_or_1);
                    }
                }
            }
        }

    }
}
