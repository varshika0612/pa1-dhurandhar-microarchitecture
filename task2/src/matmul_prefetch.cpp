// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"
#include <algorithm>
void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
     for(int i=0;i<M;++i){
        const float* a=A+i*lda;
        _mm_prefetch((const char*)(A+(i+1)*lda),_MM_HINT_T2);
        for(int j=0;j<N;++j){
            const float* b=B+j*ldb;
            _mm_prefetch((const char*)(B+(j+1*ldb)),_MM_HINT_T1);
            float acc=0.0f;
            for(int p=0;p<K;++p){
                if((p&7)==0){
               _mm_prefetch((const char*)&a[p+32],_MM_HINT_T0);
               _mm_prefetch((const char*)&b[p+32],_MM_HINT_T0);
                }
                

            
            acc+=a[p]*b[p];
            }
            C[i*ldc+j]=acc;
        }
     }
    }

