#include "matmult.h"

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {
                for(int p = 0; p < K; p++) {
                    for(int j = 0; j < N; j++) {
                        for(int i = 0; i < M; i++) {
                            C[i * N + j] += A[i * K + p] * B[p * N + j];
                        }
                    }
                }
            }