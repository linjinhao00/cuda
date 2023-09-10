#include <cuda.h>
#include <cublas.h>

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b,  float *c)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  float *start_a = a + by * BLOCK * k;
  float *start_b = b + bx * BLOCK;
  float *end_a = start_a + k;
  float sum = 0.f;

  for (float *ptr_a = start_a, *ptr_b = start_b; ptr_a < end_a; ptr_a += BLOCK, ptr_b += BLOCK * n)
  {
    __syncthreads();

    __shared__ float shared_a[BLOCK][BLOCK];
    __shared__ float shared_b[BLOCK][BLOCK];

    shared_a[ty][tx] = ptr_a[ty * k + tx];
    shared_b[ty][tx] = ptr_b[ty * n + tx];
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk)
    {
      sum += shared_a[ty][kk] * shared_b[kk][tx];
    }
  }
  c[by * BLOCK * n + bx * BLOCK + ty * n + tx] = sum;
}

void MY_MMult(cublasContext *handle, int m, int n, int k, float *A, int stride_a, float *B, int stride_b, float *C, int stride_c)
{
  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm<BLOCK><<<grid, block>>>(m, n, k, A,  B, C);
}