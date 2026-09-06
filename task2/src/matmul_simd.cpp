// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for(int i=0;i<M;i++){
      const float *a=A+i*lda;
      for(int j=0;j<N;j++){
        const float *b=B+j*ldb;
        __m256 sum_vec= _mm256_setzero_ps();
        int p=0;
        for(;p+8<=K;p+=8){
          __m256 a_vec=_mm256_loadu_ps(a+p);
          __m256 b_vec=_mm256_loadu_ps(b+p);
          sum_vec = _mm256_add_ps(sum_vec,_mm256_mul_ps(a_vec,b_vec));
          
        }
        float temp[8];
        _mm256_storeu_ps(temp,sum_vec);
        float acc= 0.0f;
        for(int t=0;t<8;t++)acc+=temp[t];
        for(;p<K;++p){
          acc+=a[p]*b[p];
        }
        C[i*ldc+j]=acc;
      }
    }
}
