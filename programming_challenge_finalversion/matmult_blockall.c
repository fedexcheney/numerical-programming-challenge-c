#include "/opt/homebrew/opt/libomp/include/omp.h"
#include "matmult.h"

#define A_lda(i, j) A[(i)*K + (j)]
#define B_lda(i, j) B[(i)*N + (j)]
#define C_lda(i, j) C[(i)*N + (j)]
#define A_blda(i, j) A_block[(i)*BLOCK_SIZE + (j)]
#define B_blda(i, j) B_block[(i)*BLOCK_SIZE + (j)]
#define C_blda(i, j) C_block[(i)*BLOCK_SIZE + (j)]
#define BLOCK_SIZE 64
#define min(x, y) ((x) < (y) ? (x) : (y))

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      int ib = min(M - ii, BLOCK_SIZE);
      int jb = min(N - jj, BLOCK_SIZE);

      double A_block[BLOCK_SIZE * BLOCK_SIZE];
      double B_block[BLOCK_SIZE * BLOCK_SIZE];
      double C_block[BLOCK_SIZE * BLOCK_SIZE];

      for (int pp = 0; pp < K; pp += BLOCK_SIZE) {
        int pb = min(K - pp, BLOCK_SIZE);

        /* pack A: A_block[i * pb + p] = A[ii+i][pp+p] (i-major, p-minor) */
        for (int i = 0; i < ib; i++) {
          for (int p = 0; p < pb; p++) {
            A_blda(i, p) = A_lda(ii + i, pp + p);
          }
        }

        /* pack B: B_block[j * pb + p] = B[pp+p][jj+j] (j-major, p-minor) */
        for (int j = 0; j < jb; j++) {
          for (int p = 0; p < pb; p++) {
            B_blda(j, p) = B_lda(pp + p, jj + j);
          }
        }

        /* pack C: C_block[i * jb + j] = C[ii+i][jj+j] (i-major, j-minor) */
        for (int i = 0; i < ib; i++) {
          for (int j = 0; j < jb; j++) {
            C_blda(i, j) = C_lda(ii + i, jj + j);
          }
        }

        for (int i = 0; i < ib; i += 2) {
          for (int j = 0; j < jb; j += 2) {
            register double c00 = C_lda(i, j);
            register double c01 = C_lda(i, (j + 1));
            register double c10 = C_lda((i + 1), j);
            register double c11 = C_lda((i + 1), (j + 1));

            // #pragma omp unroll partial(4)
            for (int p = 0; p < pb; p++) {
              register double a0 = A_lda(i, p);
              register double a1 = A_lda((i + 1), p);
              register double b0 = B_blda(j, p);
              register double b1 = B_blda((j + 1), p);

              c00 += a0 * b0;
              c01 += a0 * b1;
              c10 += a1 * b0;
              c11 += a1 * b1;
            }

            C_blda(i, j) = c00;
            C_blda(i, (j + 1)) = c01;
            C_blda((i + 1), j) = c10;
            C_blda((i + 1), (j + 1)) = c11;
          }
        }
      }
      /* unpack C_block back to C */
      for (int i = 0; i < ib; i++) {
        for (int j = 0; j < jb; j++) {
          C_lda(ii + i, jj + j) = C_blda(i, j);
        }
      }
    }
  }
}