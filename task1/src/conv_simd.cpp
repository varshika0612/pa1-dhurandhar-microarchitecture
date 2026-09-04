// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K/2;
    const int in_stride = W + 2*p;
    
    for(int i = 0; i < H*W; ++i) out[i] = 0.0f;

    for(int oy=0; oy<H; ++oy){
        int ox = 0;
        for(;ox+8<=W;ox+=8){
            __m256 outv=_mm256_setzero_ps();
            for(int ky=0;ky<K;++ky){
                for(int kx=0;kx<K;++kx){
                    __m256 inv=_mm256_loadu_ps(&in[(oy+ky)*in_stride+(ox+kx)]);
                    __m256 kv=_mm256_set1_ps(ker[ky*K+kx]);
                    outv=_mm256_fmadd_ps(inv,kv,outv);
                }
            }
            _mm256_storeu_ps(&out[oy*W+ox],outv);
        }
        for(;ox<W;++ox){
            float acc=0.0f;
            for(int ky=0;ky<K;++ky){
                for(int kx = 0; kx <K; ++kx){
                    acc+=in[(oy+ky)*in_stride+(ox+kx)]*ker[ky*K+kx];
                }
                out[oy*W+ox]=acc;
            }
        }
    }
}
