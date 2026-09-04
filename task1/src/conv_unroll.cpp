// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p=K/2;
    const int in_stride=W+2*p;

    if(K == 3) {
        float* out_ptr = out;
        for(int oy = 0; oy < H; ++oy) {
            const float* in_ptr = in + oy * in_stride;
            for(int ox = 0; ox < W; ++ox, ++out_ptr, ++in_ptr){
                float acc = 0.0f;

                // Row 0
                acc += in_ptr[0] * ker[0];
                acc += in_ptr[1] * ker[1];
                acc += in_ptr[2] * ker[2];

                // Row 1
                acc += in_ptr[in_stride + 0] * ker[3];
                acc += in_ptr[in_stride + 1] * ker[4];
                acc += in_ptr[in_stride + 2] * ker[5];

                // Row 2
                acc += in_ptr[2 * in_stride + 0] * ker[6];
                acc += in_ptr[2 * in_stride + 1] * ker[7];
                acc += in_ptr[2 * in_stride + 2] * ker[8];

                *out_ptr = acc;
            }
        }
        return;
    }

    else {
        float* out_ptr = out;
        for(int oy = 0; oy < H; ++oy){
            const float* in_ptr = in + oy * in_stride;
            for(int ox = 0; ox < W; ++ox, ++out_ptr, ++in_ptr){
                float acc = 0.0f;

                //Row 0
                acc += in_ptr[0] * ker[0];
                acc += in_ptr[1] * ker[1];
                acc += in_ptr[2] * ker[2];
                acc += in_ptr[3] * ker[3];
                acc += in_ptr[4] * ker[4];

                //Row 1
                acc += in_ptr[in_stride + 0] * ker[5];
                acc += in_ptr[in_stride + 1] * ker[6];
                acc += in_ptr[in_stride + 2] * ker[7];
                acc += in_ptr[in_stride + 3] * ker[8];
                acc += in_ptr[in_stride + 4] * ker[9];

                //Row 2
                acc += in_ptr[2 * in_stride + 0] * ker[10];
                acc += in_ptr[2 * in_stride + 1] * ker[11];
                acc += in_ptr[2 * in_stride + 2] * ker[12];
                acc += in_ptr[2 * in_stride + 3] * ker[13];
                acc += in_ptr[2 * in_stride + 4] * ker[14];

                //Row 3
                acc += in_ptr[3 * in_stride + 0] * ker[15];
                acc += in_ptr[3 * in_stride + 1] * ker[16];
                acc += in_ptr[3 * in_stride + 2] * ker[17];
                acc += in_ptr[3 * in_stride + 3] * ker[18];
                acc += in_ptr[3 * in_stride + 4] * ker[19];

                //Row 4
                acc += in_ptr[4 * in_stride + 0] * ker[20];
                acc += in_ptr[4 * in_stride + 1] * ker[21];
                acc += in_ptr[4 * in_stride + 2] * ker[22];
                acc += in_ptr[4 * in_stride + 3] * ker[23];
                acc += in_ptr[4 * in_stride + 4] * ker[24];

                *out_ptr = acc;
            }
        }

        return;
    }
}

