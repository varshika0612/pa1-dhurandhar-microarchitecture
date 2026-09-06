// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include "matmul.h"
#include <algorithm>

using namespace std;

static inline float opt_add(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v);
    __m128 hi=_mm256_extractf128_ps(v,1);
    __m128 sum=_mm_add_ps(lo,hi);
    sum=_mm_hadd_ps(sum,sum);
    sum=_mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(sum);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    constexpr int BN=128;
    constexpr int BM=64;

    for(int bi =0 ; bi<M; bi+=BM){
        
        int i_max = min(bi+BM,M);

        for(int bj=0; bj<N; bj+=BN){

            int i = bi; 

            int j_max = min(bj+BN,N);


            for(;i+3<i_max; i+=4){
                const float*a0 = A + static_cast<long> (i) * lda;
                const float* a1 =A + static_cast<long> (i+1)*lda;
                const float* a2 = A + static_cast<long> (i+2)*lda;
                const float* a3 = A + static_cast<long>(i+3)*lda;

                int j= bj;

                for(;j+1<j_max; j+=2){
                    const float* b0 = B+ static_cast<long>(j)*ldb;
                    const  float*b1= B+static_cast<long>  (j+1)*ldb;

                    __m256  c00 = _mm256_setzero_ps();
                    __m256  c01= _mm256_setzero_ps();
                    __m256  c10 =_mm256_setzero_ps();
                    __m256  c11 = _mm256_setzero_ps();
                    __m256 c20= _mm256_setzero_ps();
                    __m256  c21 =_mm256_setzero_ps();
                    __m256 c30 = _mm256_setzero_ps();
                    __m256  c31 =  _mm256_setzero_ps();

                    int p=0;

                    for( ;p+7 <K; p+= 8){
                        _mm_prefetch( reinterpret_cast< const char* > (b0 +p+ 32), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast  <const char*>(b1 +p+32),_MM_HINT_T0);

                        __m256 vb1=_mm256_loadu_ps(b1+p);
                        __m256 vb0=_mm256_loadu_ps(b0+p);

                        __m256 va0 = _mm256_loadu_ps(a0+p);
                        c00 = _mm256_fmadd_ps(va0, vb0, c00);

                        c01=_mm256_fmadd_ps(va0,vb1,c01);

                        __m256 va1 = _mm256_loadu_ps(a1 +p);

                            
                            c10=_mm256_fmadd_ps(va1,vb0,c10);
                        
                        c11=_mm256_fmadd_ps(va1,vb1,c11);

                        __m256 va2=_mm256_loadu_ps(a2+p);
                        
                        c20=_mm256_fmadd_ps(va2,vb0,c20);
                            c21=_mm256_fmadd_ps(va2,vb1,c21);

                        __m256 va3=_mm256_loadu_ps(a3+p);
                            c30=_mm256_fmadd_ps(va3,vb0,c30);

                        c31=_mm256_fmadd_ps(va3,vb1,c31);
                    }

                                        float acc00=opt_add(c00);
                    float acc01=  opt_add (c01);
                    float acc10 =opt_add(c10);
                    float acc11= opt_add(c11);

                    float acc21= opt_add(c21);

                    float acc20=opt_add(c20);

                    float acc30= opt_add  (c30);
                    float acc31=opt_add(c31);

                    for(; p< K; ++p){
                        float  bp0 = b0[p],  bp1= b1[p];

                        acc00+=a0[p]* bp0; acc01+=a0[p]*bp1;
                        acc10+=a1[p]*bp0; acc11 +=a1[p]*bp1;

                        acc20+=a2[p]*bp0; acc21 +=a2[p]*bp1;
                        acc30+=a3[p] *bp0; acc31+=a3[p]*bp1;

                    }

                    C[static_cast<long>(i)*ldc+j]=acc00;
                    C[static_cast<long>(i) *ldc+j+1]=acc01;
                    C[static_cast<long>(i+1)* ldc+j]=acc10;
                    C[static_cast<long>( i+1)*ldc+j+1]=acc11;
                    C[static_cast<long>(i+2) *ldc+j]=acc20;
                    C[static_cast<long> (i+2)*ldc+j+1]=acc21;
                    C[static_cast<long>(i+3)*ldc+j] =acc30;
                    C[static_cast<long> (i+3)*ldc+j+1]= acc31;
                }

                for(; j<j_max; ++j){
                    const float * b=B +static_cast<long> (j)*ldb;
                    for( int r =0; r<4; ++r){
                        const float * ar = A+ static_cast< long> (i+r)* lda;
                        int p = 0;
                        float acc = 0.0f;

                        __m256 vacc = _mm256_setzero_ps();

                        for(; p+7 < K; p+=8){
                            vacc  = _mm256_fmadd_ps (_mm256_loadu_ps(ar+p), _mm256_loadu_ps(b+p), vacc);

                        }

                        acc  = opt_add(vacc);

                        for(; p<K; ++p){
                            acc+=ar[p]*b[p];
                        }

                        C[static_cast <long> (i+r)* ldc+j] = acc;
                    }
                }
            }


            for(;  i<i_max; i++){
                const float  * a= A + static_cast< long>  (i)  * lda;

                for( int j= bj; j<j_max; j++){
                    const float * b = B + static_cast <long> (j) * ldb;
                    int p=0;
                    __m256 vacc = _mm256_setzero_ps();
                    float acc = 0.0f;

                    for(;  p+7 <K; p+=8){
                        vacc = _mm256_fmadd_ps (_mm256_loadu_ps(a+p),_mm256_loadu_ps(b+p),vacc);
                    }

                    acc = opt_add(vacc);

                    for(;p<K;++p) acc+=a[p]*b[p];
                    C[static_cast<long>(i)*ldc+j]=acc;
                }
            }
        }
    }
}
