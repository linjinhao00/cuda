#include <cublas.h>
#include <cuda.h>

template <int BLOCK, int STRDIE>
__global__ void sgemm(int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  constexpr int STEP = BLOCK * STRDIE;

  float *begin_a = a + STEP * k * by;
  float *begin_b = b + STEP * bx;
  float *end_a = begin_a + k;

  float sum[STRDIE][STRDIE] = {0.f};

  for (float *ptr_a = begin_a, *ptr_b = begin_b; ptr_a < end_a; ptr_a += STEP, ptr_b += STEP * n)
  {
    __shared__ float share_a[STEP][STEP];
    __shared__ float share_b[STEP][STEP];

    for (int i = 0; i < STRDIE; ++i)
    {
      for (int j = 0; j < STRDIE; ++j)
      {
        share_a[ty * STRDIE + i][tx * STRDIE + j] = ptr_a[(ty * STRDIE + i) * k + tx * STRDIE + j];
        share_b[ty * STRDIE + i][tx * STRDIE + j] = ptr_b[(ty * STRDIE + i) * n + tx * STRDIE + j];
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < STRDIE; ++i)
    {
      for (int j = 0; j < STRDIE; ++j)
      {
        for (int kk = 0; kk < STEP; ++kk)
        {
          sum[i][j] += share_a[ty * STRDIE + i][kk] * share_b[kk][tx * STRDIE + j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < STRDIE; ++i)
  {
    for (int j = 0; j < STRDIE; ++j)
    {
      c[(by * STEP + ty * STRDIE + i) * n + bx * STEP + tx * STRDIE + j] = sum[i][j];
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