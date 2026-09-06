// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K/2;
    const int in_stride = W + 2*p;

    if(K == 3){
        __m256 k0 = _mm256_set1_ps(ker[0]);
        __m256 k1 = _mm256_set1_ps(ker[1]);
        __m256 k2 = _mm256_set1_ps(ker[2]);
        __m256 k3 = _mm256_set1_ps(ker[3]);
        __m256 k4 = _mm256_set1_ps(ker[4]);
        __m256 k5 = _mm256_set1_ps(ker[5]);
        __m256 k6 = _mm256_set1_ps(ker[6]);
        __m256 k7 = _mm256_set1_ps(ker[7]);
        __m256 k8 = _mm256_set1_ps(ker[8]);

        for(int oy = 0; oy < H; ++oy){
            int ox=0;

            for(;ox + 8 <= W; ox += 8){
                __m256 outv = _mm256_setzero_ps();

                // Row 0
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 0)]), k0, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 1)]), k1, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 2)]), k2, outv);

                // Row 1
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 0)]), k3, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 1)]), k4, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 2)]), k5, outv);

                // Row 2
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 0)]), k6, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 1)]), k7, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 2)]), k8, outv);

                _mm256_storeu_ps(&out[oy*W + ox], outv);
            }

            for(; ox < W; ++ox){
                float acc = 0.0f;

                // Row 0
                acc += in[(oy + 0) * in_stride + (ox + 0)] * ker[0];
                acc += in[(oy + 0) * in_stride + (ox + 1)] * ker[1];
                acc += in[(oy + 0) * in_stride + (ox + 2)] * ker[2];

                // Row 1
                acc += in[(oy + 1) * in_stride + (ox + 0)] * ker[3];
                acc += in[(oy + 1) * in_stride + (ox + 1)] * ker[4];
                acc += in[(oy + 1) * in_stride + (ox + 2)] * ker[5];

                // Row 2
                acc += in[(oy + 2) * in_stride + (ox + 0)] * ker[6];
                acc += in[(oy + 2) * in_stride + (ox + 1)] * ker[7];
                acc += in[(oy + 2) * in_stride + (ox + 2)] * ker[8];

                out[oy * W + ox] = acc;
            }
        }
    }

    else if (K == 5){
        __m256 k0 = _mm256_set1_ps(ker[0]);
        __m256 k1 = _mm256_set1_ps(ker[1]);
        __m256 k2 = _mm256_set1_ps(ker[2]);
        __m256 k3 = _mm256_set1_ps(ker[3]);
        __m256 k4 = _mm256_set1_ps(ker[4]);
        __m256 k5 = _mm256_set1_ps(ker[5]);
        __m256 k6 = _mm256_set1_ps(ker[6]);
        __m256 k7 = _mm256_set1_ps(ker[7]);
        __m256 k8 = _mm256_set1_ps(ker[8]);
        __m256 k9  = _mm256_set1_ps(ker[9]);
        __m256 k10 = _mm256_set1_ps(ker[10]);
        __m256 k11 = _mm256_set1_ps(ker[11]);
        __m256 k12 = _mm256_set1_ps(ker[12]);
        __m256 k13 = _mm256_set1_ps(ker[13]);
        __m256 k14 = _mm256_set1_ps(ker[14]);
        __m256 k15 = _mm256_set1_ps(ker[15]);
        __m256 k16 = _mm256_set1_ps(ker[16]);
        __m256 k17 = _mm256_set1_ps(ker[17]);
        __m256 k18 = _mm256_set1_ps(ker[18]);
        __m256 k19 = _mm256_set1_ps(ker[19]);
        __m256 k20 = _mm256_set1_ps(ker[20]);
        __m256 k21 = _mm256_set1_ps(ker[21]);
        __m256 k22 = _mm256_set1_ps(ker[22]);
        __m256 k23 = _mm256_set1_ps(ker[23]);
        __m256 k24 = _mm256_set1_ps(ker[24]);

        for(int oy=0; oy<H; ++oy){
            int ox=0;

            for(; ox + 8 <= W; ox += 8){
                __m256 outv=_mm256_setzero_ps();

                // Row 0
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 0)]), k0, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 1)]), k1, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 2)]), k2, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 3)]), k3, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 4)]), k4, outv);

                // Row 1
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 0)]), k5, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 1)]), k6, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 2)]), k7, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 3)]), k8, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 4)]), k9, outv);

                // Row 2
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 0)]), k10, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 1)]), k11, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 2)]), k12, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 3)]), k13, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 4)]), k14, outv);

                // Row 3
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + 0)]), k15, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + 1)]), k16, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + 2)]), k17, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + 3)]), k18, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + 4)]), k19, outv);

                // Row 4
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 4) * in_stride + (ox + 0)]), k20, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 4) * in_stride + (ox + 1)]), k21, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 4) * in_stride + (ox + 2)]), k22, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 4) * in_stride + (ox + 3)]), k23, outv);
                outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 4) * in_stride + (ox + 4)]), k24, outv);

                _mm256_storeu_ps(&out[oy * W + ox], outv);
            }

            for(; ox < W; ++ox) {
                float acc = 0.0f;

                // Row 0
                acc += in[(oy + 0) * in_stride + (ox + 0)] * ker[0];
                acc += in[(oy + 0) * in_stride + (ox + 1)] * ker[1];
                acc += in[(oy + 0) * in_stride + (ox + 2)] * ker[2];
                acc += in[(oy + 0) * in_stride + (ox + 3)] * ker[3];
                acc += in[(oy + 0) * in_stride + (ox + 4)] * ker[4];

                // Row 1
                acc += in[(oy + 1) * in_stride + (ox + 0)] * ker[5];
                acc += in[(oy + 1) * in_stride + (ox + 1)] * ker[6];
                acc += in[(oy + 1) * in_stride + (ox + 2)] * ker[7];
                acc += in[(oy + 1) * in_stride + (ox + 3)] * ker[8];
                acc += in[(oy + 1) * in_stride + (ox + 4)] * ker[9];

                // Row 2
                acc += in[(oy + 2) * in_stride + (ox + 0)] * ker[10];
                acc += in[(oy + 2) * in_stride + (ox + 1)] * ker[11];
                acc += in[(oy + 2) * in_stride + (ox + 2)] * ker[12];
                acc += in[(oy + 2) * in_stride + (ox + 3)] * ker[13];
                acc += in[(oy + 2) * in_stride + (ox + 4)] * ker[14];

                // Row 3
                acc += in[(oy + 3) * in_stride + (ox + 0)] * ker[15];
                acc += in[(oy + 3) * in_stride + (ox + 1)] * ker[16];
                acc += in[(oy + 3) * in_stride + (ox + 2)] * ker[17];
                acc += in[(oy + 3) * in_stride + (ox + 3)] * ker[18];
                acc += in[(oy + 3) * in_stride + (ox + 4)] * ker[19];

                // Row 4
                acc += in[(oy + 4) * in_stride + (ox + 0)] * ker[20];
                acc += in[(oy + 4) * in_stride + (ox + 1)] * ker[21];
                acc += in[(oy + 4) * in_stride + (ox + 2)] * ker[22];
                acc += in[(oy + 4) * in_stride + (ox + 3)] * ker[23];
                acc += in[(oy + 4) * in_stride + (ox + 4)] * ker[24];

                out[oy * W + ox] = acc;
            }
        }
    }

    else {
        for(int oy = 0; oy < H; ++oy) {
            int ox = 0;

            for(; ox + 8 <= W; ox += 8) {
                __m256 outv = _mm256_setzero_ps();

                for(int ky = 0; ky < K; ++ky) {
                    for(int kx = 0; kx < K; ++kx) {
                        __m256 inv = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                        __m256 kv = _mm256_set1_ps(ker[ky * K + kx]);
                        outv = _mm256_fmadd_ps(inv, kv, outv);
                    }
                }

                _mm256_storeu_ps(&out[oy * W + ox], outv);
            }

            for(; ox < W; ++ox) {
                float acc = 0.0f;

                for(int ky = 0; ky < K; ++ky) {
                    for(int kx = 0; kx < K; ++kx) {
                        acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    }
                }

                out[oy * W + ox] = acc;
            }
        }
    }
}
