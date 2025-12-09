#include "matmult.h"
#include <Accelerate/Accelerate.h>
// https://developer.apple.com/documentation/accelerate/cblas_dgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)
void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {

  // cblas_dgemm Double-precision GEneral Matrix Multiply
  // C = alpha * (op(A) * op(B)) + beta * C

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0, A,
              (int)K, // lda: A
              B,
              (int)N, // ldb: B
              0.0, C,
              (int)N // ldc: C
  );
}