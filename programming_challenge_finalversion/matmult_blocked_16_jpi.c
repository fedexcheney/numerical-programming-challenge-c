#include "matmult.h"

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C)
{
    const int64_t BLOCK_SIZE = 16;
    for (int jj = 0; jj < N; jj += BLOCK_SIZE)
    {
        for (int pp = 0; pp < K; pp += BLOCK_SIZE)
        {
            for (int ii = 0; ii < M; ii += BLOCK_SIZE)
            {
                for (int j = jj; j < (jj + BLOCK_SIZE) && j < N; j++)
                {
                    for (int p = pp; p < (pp + BLOCK_SIZE) && p < K; p++)
                    {
                        for (int i = ii; i < (ii + BLOCK_SIZE) && i < N; i++)
                        {
                            C[i * N + j] += A[i * K + p] * B[p * N + j];
                        }
                    }
                }
            }
        }
    }
}