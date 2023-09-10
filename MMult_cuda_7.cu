#include <cuda.h>
#include <cublas.h>

template <int BLOCK, int STRIDE>
__global__ void sgemm(int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y * 2;
    const int tx = threadIdx.x * 2;
    constexpr int STEP = BLOCK * STRIDE * 2;

    float *abegin = a + by * STEP * k;
    float *bbegin = b + bx * STEP;
    float *aend = abegin + k;

    __shared__ float ashare[STEP][STEP];
    __shared__ float bshare[STEP][STEP];
    float sum0[STRIDE][STRIDE] = {0.f};
    float sum1[STRIDE][STRIDE] = {0.f};
    float sum2[STRIDE][STRIDE] = {0.f};
    float sum3[STRIDE][STRIDE] = {0.f};

    for (float *aptr = abegin, *bptr = bbegin; aptr < aend; aptr += STEP, bptr += STEP * n)
    {

        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                ashare[ty + i][tx + j] = aptr[(ty + i) * k + tx + j];
                ashare[ty + i][tx + j + 32] = aptr[(ty + i) * k + tx + j + 32];
                ashare[ty + i + 32][tx + j] = aptr[(ty + i + 32) * k + tx + j];
                ashare[ty + i + 32][tx + j + 32] = aptr[(ty + i + 32) * k + j + tx + 32];

                bshare[ty + i][tx + j] = bptr[(ty + i) * n + tx + j];
                bshare[ty + i][tx + j + 32] = bptr[(ty + i) * n + tx + j + 32];
                bshare[ty + i + 32][tx + j] = bptr[(ty + i + 32) * n + tx + j];
                bshare[ty + i + 32][tx + j + 32] = bptr[(ty + i + 32) * n + tx + j + 32];
            }
        }
        __syncthreads();

        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                for (int kk = 0; kk < STEP; ++kk)
                {
                    sum0[i][j] += ashare[ty + i][kk] * bshare[kk][tx + j];
                    sum1[i][j] += ashare[ty + i][kk] * bshare[kk][tx + j + 32];
                    sum2[i][j] += ashare[ty + i + 32][kk] * bshare[kk][tx + j];
                    sum3[i][j] += ashare[ty + i + 32][kk] * bshare[kk][tx + j + 32];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            c[(by * STEP + ty + i) * n + bx * STEP + tx + j] = sum0[i][j];
            c[(by * STEP + ty + i) * n + bx * STEP + tx + 32 + j] = sum1[i][j];
            c[(by * STEP + ty + 32 + i) * n + bx * STEP + tx + j] = sum2[i][j];
            c[(by * STEP + ty + 32 + i) * n + bx * STEP + tx + 32+ j] = sum3[i][j];
        }
    }
}

void MY_MMult(cublasContext *handle, int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid(m / (BLOCK * STRIDE * 2), n / (BLOCK * STRIDE * 2));
    sgemm<BLOCK, STRIDE><<<grid, block>>>(n, m, k, a, len_a, b, len_b, c, len_c);
}