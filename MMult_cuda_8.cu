#include <cublas.h>
#include <cuda.h>

template <int BLOCK, int STRIDE>
__global__ void sgemm(int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    constexpr int STEP = BLOCK * STRIDE;

    float *abegin = a + by * STEP * k;
    float *bbegin = b + bx * STEP;
    float *aend = abegin + STEP;

    __shared__ float ashare[2][STEP][STEP];
    __shared__ float bshare[2][STEP][STEP];
    float sum[STRIDE][STRIDE] = {0.f};
    float *aptr = abegin;
    float *bptr = bbegin;
#define LOAD(idx)                                                                                              \
    do                                                                                                         \
    {                                                                                                          \
        for (int i = 0; i < STRIDE; ++i)                                                                       \
        {                                                                                                      \
            for (int j = 0; j < STRIDE; ++j)                                                                   \
            {                                                                                                  \
                ashare[idx][ty * STRIDE + i][tx * STRIDE + j] = aptr[(ty * STRIDE + i) * k + tx * STRIDE + j]; \
                bshare[idx][ty * STRIDE + i][tx * STRIDE + j] = bptr[(ty * STRIDE + i) * n + tx * STRIDE + j]; \
            }                                                                                                  \
        }                                                                                                      \
        aptr += STEP;                                                                                          \
        bptr += STEP * n;                                                                                      \
    } while (0)

#define SUM(idx)                                                                                      \
    do                                                                                                \
    {                                                                                                 \
        for (int i = 0; i < STRIDE; ++i)                                                              \
        {                                                                                             \
            for (int j = 0; j < STRIDE; ++j)                                                          \
            {                                                                                         \
                for (int kk = 0; kk < STEP; ++kk)                                                     \
                    sum[i][j] += ashare[idx][ty * STRIDE + i][kk] * bshare[idx][kk][tx * STRIDE + j]; \
            }                                                                                         \
        }                                                                                             \
    } while (0)

    LOAD(0);
    for (; aptr < aend;)
    {
        __syncthreads();
        SUM(0);
        LOAD(1);

        __syncthreads();
        if (aptr < aend)
        {
            LOAD(0);
        }
        SUM(1);
    }
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            c[(by * STEP + ty * STRIDE + i) * n + bx * STEP + tx * STRIDE + j] = sum[i][j];
        }
    }
}

void MY_MMult(cublasContext *handle, int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
    sgemm<BLOCK, STRIDE><<<grid, block>>>(n, m, k, a, len_a, b, len_b, c, len_c);
}