#include "matmult.h"
#include <arm_neon.h>

#define A_lda(i, j) A[(i)*K + (j)]
#define B_lda(i, j) B[(i)*N + (j)]
#define C_lda(i, j) C[(i)*N + (j)]

#define min(x, y) ((x) < (y) ? (x) : (y))
// vdupq_n_f64 -> vld1q_f64 -> vfmaq_f64 -> vst1q_f64
// create -> load -> perform(FMA) -> store
void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {
  const int64_t BLOCK_SIZE = 64;
  // double B_block[BLOCK_SIZE * BLOCK_SIZE];
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      C_lda(i, j) = 0.0;
    }
  }
  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    int ib = min(M - ii, BLOCK_SIZE);
    for (int pp = 0; pp < K; pp += BLOCK_SIZE) {
      int pb = min(K - pp, BLOCK_SIZE);
      for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
        int jb = min(N - jj, BLOCK_SIZE);
        // for (int p = 0; p < pb; p++) {
        //   for (int j = 0; j < jb; j++) {
        //     B_block[p * jb + j] = B_lda(pp + p, jj + j);
        //   }
        // }
        for (int64_t i = 0; i < ib; i += 2) {
          for (int64_t j = 0; j < jb; j += 2) {
            /* load C block once */
            float64x2_t c0 = vld1q_f64(&C_lda(ii + i, jj + j));     // c00, c01
            float64x2_t c1 = vld1q_f64(&C_lda(ii + i + 1, jj + j)); // c10, c11

            for (int64_t p = 0; p < pb; p++) {
              /* broadcast A scalars into vectors */
              float64x2_t a0 = vdupq_n_f64(A_lda(ii + i, pp + p));
              float64x2_t a1 = vdupq_n_f64(A_lda(ii + i + 1, pp + p));
              float64x2_t b = vld1q_f64(&B_lda(pp + p, jj + j));
              // float64x2_t b = vld1q_f64(&B_block[p * jb + j]);

              /* accumulate into C vectors */
              c0 = vfmaq_f64(c0, a0, b);
              c1 = vfmaq_f64(c1, a1, b);
            }

            /* store accumulated C block */
            vst1q_f64(&C_lda(ii + i, jj + j), c0);
            vst1q_f64(&C_lda(ii + i + 1, jj + j), c1);
          }
        }
      }
    }
  }
}