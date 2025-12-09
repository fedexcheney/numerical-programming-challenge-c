#include "/opt/homebrew/opt/libomp/include/omp.h"
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
#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      int ib = min(M - ii, BLOCK_SIZE);
      int jb = min(N - jj, BLOCK_SIZE);
      for (int pp = 0; pp < K; pp += BLOCK_SIZE) {
        int pb = min(K - pp, BLOCK_SIZE);

        for (int i = 0; i < ib; i++) {
          double *const restrict Ci =
              &C_lda(ii + i, jj); /* pointer to C[ii+i][jj:] */
          for (int p = 0; p < pb; p++) {
            const double aik = A_lda(ii + i, pp + p); /* scalar A[ii+i][pp+p] */
            const double *const restrict Bp =
                &B_lda(pp + p, jj); /* pointer to B[pp+p][jj:] */

            const float64x2_t aik_vec = vdupq_n_f64(aik);

            for (int j = 0; j < jb; j += 2) {
              float64x2_t c = vld1q_f64(&Ci[j]); /* load C[ii+i][jj+j:jj+j+2] */
              const float64x2_t b =
                  vld1q_f64(&Bp[j]);        /* load B[pp+p][jj+j:jj+j+2] */
              c = vfmaq_f64(c, aik_vec, b); /* C += aik * B */
              vst1q_f64(&Ci[j], c);         /* store back */
            }
          }
        }
      }
    }
  }
}