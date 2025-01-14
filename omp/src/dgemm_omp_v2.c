#include "omp.h"
#include "dgemm_omp.h"

#define BM 8
#define BN 24

#define DEFAULT_TILE_M 8192
#define DEFAULT_TILE_N 192
#define DEFAULT_TILE_K 384

static const int Abuf_len = 2 * DEFAULT_TILE_M * DEFAULT_TILE_K;
static const int Bbuf_len = 2 * DEFAULT_TILE_N * DEFAULT_TILE_K;

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

        _mm512_store_pd(C + i * ldc + 0,  out[0]);
        _mm512_store_pd(C + i * ldc + 8,  out[1]);
        _mm512_store_pd(C + i * ldc + 16, out[2]);
    }
}

static void do_packA_bm(double *Abuf, int bm, int Kt, int Mb, int lda, const double *Ap)
{
    for (int k = 0; k < Kt; k++)
        for (int i = 0; i < Mb; i++)
            Abuf[k * Mb + i] = Ap[(bm + i) * lda + k];
}

static void do_packB_bn(double *Bbuf, int bn, int Kt, int Nb, int ldb, const double *Bp)
{
    for (int k = 0; k < Kt; k++)
        for (int j = 0; j < Nb; j++)
            Bbuf[k * Nb + j] = Bp[k * ldb + bn + j];
}

static void decide_tile_size(int M, int N, int K, int *tile_m, int *tile_n, int *tile_k)
{
    *tile_m = DEFAULT_TILE_M;
    *tile_n = DEFAULT_TILE_N;
    *tile_k = DEFAULT_TILE_K;
}

static void check_input(int M, int N) {
    if (M % BM != 0 || N % BN != 0) {
        printf("Error: M, N must be a multiple of (%d, %d).\n", BM, BN);
        exit(-1);
    }
}

void dgemm_omp_v2(DGEMM_FUNC_SIGNITURE)
{
    check_input(M, N);

    int TILE_M, TILE_N, TILE_K;
    decide_tile_size(M, N, K, &TILE_M, &TILE_N, &TILE_K);

    double Abuf[Abuf_len] __attribute__((aligned(64)));

    
    // tiling
    for (int tm = 0; tm < M; tm += TILE_M) {
        for (int tk = 0; tk < K; tk += TILE_K) {
            // the loop tn=0 is responsable for pack A
            {
                double Bbuf_tn0[Bbuf_len] __attribute__((aligned(64)));

                int tn = 0;
                const double *Ap = A + tm * lda + tk;
                const double *Bp = B + tk * ldb + tn;
                double *Cp = C + tm * ldc + tn;

                int Mt = MIN(TILE_M, M - tm);
                int Nt = MIN(TILE_N, N - tn);
                int Kt = MIN(TILE_K, K - tk);

                #pragma omp parallel for
                for (int bn = 0; bn < Nt; bn += BN) {
                    int Nb = MIN(BN, Nt - bn);
                    int Bbuf_offset = bn * Kt;
                    do_packB_bn(Bbuf_tn0 + Bbuf_offset, bn, Kt, Nb, ldb, Bp);
                }

                #pragma omp parallel for
                for (int bm = 0; bm < Mt; bm += BM) {
                    int Mb = MIN(BM, Mt - bm);
                    int Abuf_offset = bm * Kt;

                    do_packA_bm(Abuf + Abuf_offset, bm, Kt, Mb, lda, Ap);

                    for (int bn = 0; bn < Nt; bn += BN) {
                        int Nb = MIN(BN, Nt - bn);
                        int Bbuf_offset = bn * Kt;

                        double beta_or_1 = (tk == 0) ? beta : 1;
                        kernel_n24m8(Abuf + Abuf_offset, Bbuf_tn0 + Bbuf_offset, Cp + bm * ldc + bn, ldc, Kt, alpha, beta_or_1);
                    }
                }
                
            }

            #pragma omp parallel 
            {
                double local_Bbuf[Bbuf_len] __attribute__((aligned(64)));

                #pragma omp for
                for (int tn = TILE_N; tn < N; tn += TILE_N) {
                    const double *Ap = A + tm * lda + tk;
                    const double *Bp = B + tk * ldb + tn;
                    double *Cp = C + tm * ldc + tn;

                    int Mt = MIN(TILE_M, M - tm);
                    int Nt = MIN(TILE_N, N - tn);
                    int Kt = MIN(TILE_K, K - tk);

                    for (int bm = 0; bm < Mt; bm += BM) {
                        for (int bn = 0; bn < Nt; bn += BN) {
                            int Mb = MIN(BM, Mt - bm);
                            int Abuf_offset = bm * Kt;

                            int Nb = MIN(BN, Nt - bn);
                            int Bbuf_offset = bn * Kt;
                            if (bm == 0)
                                do_packB_bn(local_Bbuf + Bbuf_offset, bn, Kt, Nb, ldb, Bp);
    
                            double beta_or_1 = (tk == 0) ? beta : 1;
                            kernel_n24m8(Abuf + Abuf_offset, local_Bbuf + Bbuf_offset, Cp + bm * ldc + bn, ldc, Kt, alpha, beta_or_1);
                        }
                    }
                }
            }

        }

    }
}
