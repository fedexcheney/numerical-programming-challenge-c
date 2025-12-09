#include "matmult.h"
// Assume C = M x N, A = M x K, B = K x N
void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {
                for(int i = 0; i < M; i++){
                    for(int j = 0; j < N; j++){
                        for(int p = 0; p < K; p++){
                            C[i * N + j] += A[i * K + p] * B[p * N + j];
                        }
                    }
                }
            }