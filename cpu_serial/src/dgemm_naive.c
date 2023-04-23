#include "dgemm.h"


void dgemm_naive(DGEMM_FUNC_SIGNITURE) 
{
    if (beta != 1)
        scale(C, M * N, beta);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C(i, j) += alpha * A(i, k) * B(k, j);
}