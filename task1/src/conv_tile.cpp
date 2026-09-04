// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include<algorithm>
#include<cstring>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    std::memset(out, 0, sizeof(float) * H * W);

    constexpr int TILE_H = 4;
    constexpr int TILE_W = 64;

    for(int oy0 = 0; oy0 < H; oy0 += TILE_H){
        const int oy_end = std::min(oy0 += TILE_H, H);

        for(int ox0 = 0; ox0 < W; ox0 += TILE_W){
            const int ox_end = std::min(ox0 + TILE_W, W);
            for(int oy = oy0; oy < oy_end; ++oy){
                for(int ox = ox0; ox < ox_end; ++ox){
                    float acc = 0.0f;
                    for(int ky = 0; ky < K; ++ky){
                        for(int kx = 0; kx < K; ++kx){
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
