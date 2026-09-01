// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p=K/2;
    const int in_stride=W+2*p;

    for(int oy=0;oy<H;++oy){
        for(int ox=0;ox<W;++ox){
            float acc=0.0f;
            for(int ky=0;ky<K;++ky){
                int kx=0;
                for(;kx+4<=K;kx+=4){
                    acc+=in[(oy+ky)*in_stride+(ox+kx)]*ker[ky*K+kx]
                         +in[(oy+ky)*in_stride+(ox+kx+1)]*ker[ky*K+(kx+1)]
                         +in[(oy+ky)*in_stride+(ox+kx+2)]*ker[ky*K+(kx+2)]
                         +in[(oy+ky)*in_stride+(ox+kx+3)]*ker[ky*K+(kx+3)];
                }
                for(;kx<K;++kx){
                    acc+=in[(oy+ky)*in_stride+(ox+kx)]*ker[ky*K+kx];

                }
                out[oy*W+ox]=acc;

                }
            }
        }
    }

