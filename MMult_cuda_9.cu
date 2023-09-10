#include <cublas.h>
#include <cuda.h>

#define SHARE_MEM_W 132

template <int BLOCK>
__global__ void sgemm(int n, int m, int k, float *a, float *b, float *c)
{
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  __shared__ char share_mem[1024 * 24];
  float *ashare = reinterpret_cast<float*>(share_mem);
  float *bshare = reinterpret_cast<float*>(share_mem + 16 * 1024);

  float sum[8][8] = {0.f};
  float aplane[8] = {0.f};
  float bplane[8] = {0.f};

  int afrom = by * 128 * k + tx / 8 * 4 * k + tx % 8;
  int bfrom = bx * 128 + tx / 32 * n + tx % 32;
  for (int kk = 0; kk < k; kk += 8)
  {
    // move from a to ashare
    int ato = tx / 8 * 4 + tx % 8 * SHARE_MEM_W;
    for (int i = 0; i < 4; ++i)
    {
      ashare[ato + i] = a[afrom + i * k];
    }
    // move from b to bshare
    int bto = tx / 32 * SHARE_MEM_W + tx % 32;
    for (int i = 0; i < 4; ++i)
    {
      bshare[bto + i * 32] = b[bfrom + i * 32];
    }
    __syncthreads();

    afrom += 8;
    bfrom += 8 * n;

    for (int i = 0; i < 8; ++i)
    {
      int a_share_from = tx / 16 * 4 + i * SHARE_MEM_W;
      for (int j = 0; j < 4; ++j)
      {
        aplane[j] = ashare[a_share_from + j];
        aplane[j + 4] = ashare[a_share_from + 64 + j];
      }

      int b_share_from = tx % 16 * 4 + i * SHARE_MEM_W;
      for (int j = 0; j < 4; ++j)
      {
        bplane[j] = bshare[b_share_from + j];
        bplane[j + 4] = bshare[b_share_from + 64 + j];
      }

      for (int j = 0; j < 8; ++j)
      {
        for (int k = 0; k < 8; ++k)
        {
          sum[j][k] += aplane[j] * bplane[k];
        }
      }
    }
    __syncthreads();
  }
  int cstart = by * 128 * n  + tx / 16 * 4 * n + bx * 128 + tx % 16 * 4;
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; ++j){
      c[cstart + i* n + j] = sum[i][j]; 
      c[cstart + i* n + j + 64] = sum[i][j + 4]; 
      c[cstart + (i + 64) * n + j] = sum[i + 4][j]; 
      c[cstart + (i + 64) * n + j + 64] = sum[i + 4][j + 4]; 
    }
  }

}

void MY_MMult(cublasContext *handle, int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
  constexpr int BLOCK = 128;
  dim3 block(BLOCK * 2);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm<BLOCK><<<grid, block>>>(n, m, k, a, b, c);
}