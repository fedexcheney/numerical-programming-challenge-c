#include "matmult.h"
#include "/opt/homebrew/opt/libomp/include/omp.h"

#define A_lda(i, j) A[(i)*K + (j)]
#define B_ldb(i, j) B[(i)*N + (j)]
#define C_ldc(i, j) C[(i)*N + (j)]
#define B_bldb(i, j) B_block[(i)*BLOCK_SIZE + (j)]
#define min(x, y) ((x) < (y) ? (x) : (y))

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {
  const int BLOCK_SIZE = 64;
  double B_block[BLOCK_SIZE * BLOCK_SIZE];
#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    int ib = min(M - ii, BLOCK_SIZE);
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      int jb = min(N - jj, BLOCK_SIZE);
      for (int pp = 0; pp < K; pp += BLOCK_SIZE) {
        int pb = min(K - pp, BLOCK_SIZE);

        for (int j = 0; j < jb; j++) {
          for (int p = 0; p < pb; p++) {
            B_bldb(j, p) = B_ldb(pp + p, jj + j); // B transpose
          }
        }

        for (int i = 0; i < ib; i += 2) {
          for (int j = 0; j < jb; j += 2) {
            register double c00 = C_ldc(ii + i, jj + j);
            register double c01 = C_ldc(ii + i, jj + j + 1);
            register double c10 = C_ldc(ii + i + 1, jj + j);
            register double c11 = C_ldc(ii + i + 1, jj + j + 1);
            for (int p = 0; p < pb; p++) {
              register double a0 = A_lda(ii + i, pp + p);
              register double a1 = A_lda(ii + i + 1, pp + p);
              register double b0 = B_bldb(j, p);
              register double b1 = B_bldb((j + 1), p);

              c00 += a0 * b0;
              c01 += a0 * b1;
              c10 += a1 * b0;
              c11 += a1 * b1;
            }
            C_ldc(ii + i, jj + j) = c00;
            C_ldc(ii + i, jj + j + 1) = c01;
            C_ldc(ii + i + 1, jj + j) = c10;
            C_ldc(ii + i + 1, jj + j + 1) = c11;
          }
        }
      }
    }
  }
}