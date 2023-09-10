#include <cublas.h>
#include <cuda.h>

#define SMEM_STRIDE 128
#define SMEM_STRIDE_C 64

__device__ __forceinline__ uint32_t smem_addr_to_u32(const void *smem_ptr)
{
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      "cvta.to.shared.u64 u64addr, %1;\n"
      "cvt.u32.u64 %0, u64addr;\n}"
      : "=r"(addr)
      : "l"(smem_ptr));
  return addr;
}

__device__ __forceinline__ void ld_global_to_reg_32(float &reg, const void *ptr)
{
  asm volatile("{.reg .pred p;\n"
               "mov.b32 %0, 0;\n"
#if __CUDA_API_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
               "ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
               "ld.global.nc.f32 %0, [%1];}\n"
#endif
               : "=f"(reg)
               : "l"(ptr));
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2, float &reg3, const uint32_t &addr)
{
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1, const float &reg2, const float &reg3, const uint32_t addr)
{
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t addr)
{
  asm volatile("st.shared.f32 [%0], %1;\n"
               :
               : "r"(addr), "f"(reg));
}

__device__ __forceinline__ void stg32(const float &reg, const void *ptr)
{
  asm volatile("{ .reg .pred p;\n"
               "st.global.f32 [%0], %1;}\n"
               :
               : "l"(ptr), "f"(reg));
}

template <int BLOCK>
__global__ __launch_bounds__(256, 2) void sgemm(int n, int m, int k, float *a, float *b, float *c)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  __shared__ __align__(16 * 1024) char smem[1024 * 24];
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare = reinterpret_cast<float *>(smem + 16 * 1024);
  float sum[8][8] = {0.f};
  float a_reg[4] = {0.f};
  float b_reg[4] = {0.f};
  float a_plane[2][8] = {0.f};
  float b_plane[2][8] = {0.f};

  int a_from = by * 128 * k + tx / 8 * 4 * k + tx % 8;
  int b_from = bx * 128 + tx / 32 * n + tx % 32;

  float *a_from_ptr = a + a_from;
  float *b_from_ptr = b + b_from;
  uint32_t a_to_ptr = smem_addr_to_u32(ashare + tx / 8 * 4 + tx % 8 * SMEM_STRIDE);
  uint32_t b_to_ptr = smem_addr_to_u32(bshare + tx / 32 * SMEM_STRIDE + tx % 32);

  uint32_t a_share_from_ptr = smem_addr_to_u32(ashare + tx / 16 * 4);
  uint32_t b_share_from_ptr = smem_addr_to_u32(bshare + tx % 16 * 4);

  // 初始化流水线
  for (int i = 0; i < 4; ++i)
  {
    ld_global_to_reg_32(a_reg[i], (const char *)(a_from_ptr) + i * k * sizeof(float));
  }
  sts128(a_reg[0], a_reg[1], a_reg[2], a_reg[3], a_to_ptr);

  for (int i = 0; i < 4; ++i)
  {
    ld_global_to_reg_32(b_reg[i], (const char *)(b_from_ptr) + i * 32 * sizeof(float));
  }
  for (int i = 0; i < 4; ++i)
  {
    sts32(b_reg[i], b_to_ptr + i * sizeof(float) * 32);
  }
  __syncthreads();

  lds128(a_plane[0][0], a_plane[0][1], a_plane[0][2], a_plane[0][3],
         a_share_from_ptr);
  lds128(a_plane[0][4], a_plane[0][5], a_plane[0][6], a_plane[0][7],
         a_share_from_ptr + 64 * sizeof(float));
  lds128(b_plane[0][0], b_plane[0][1], b_plane[0][2], b_plane[0][3],
         b_share_from_ptr);
  lds128(b_plane[0][4], b_plane[0][5], b_plane[0][6], b_plane[0][7],
         b_share_from_ptr + 64 * sizeof(float));

  a_from_ptr += 8;
  b_from_ptr += 8 * n;
  a_to_ptr ^= 0x2000;
  b_to_ptr ^= 0x1000;


  for (int kk = 0; kk < k; kk += 8)
  {
    for (int line = 0; line < 8; ++line)
    {

      if (line == 7 && kk < k - 8)
      {
        sts128(a_reg[0], a_reg[1], a_reg[2], a_reg[3], a_to_ptr);
        for (int i = 0; i < 4; ++i)
        {
          sts32(b_reg[i], b_to_ptr + i * sizeof(float) * 32);
        }
        __syncthreads();
        a_from_ptr += 8;
        b_from_ptr += 8 * n;
        a_to_ptr ^= 0x2000;
        b_to_ptr ^= 0x1000;
        a_share_from_ptr ^= 0x2000;
        b_share_from_ptr ^= 0x1000;
      }

      int pp = (line + 1) % 2;
      lds128(a_plane[pp][0], a_plane[pp][1], a_plane[pp][2], a_plane[pp][3],
             a_share_from_ptr + (line + 1) % 8 * SMEM_STRIDE * sizeof(float));
      lds128(a_plane[pp][4], a_plane[pp][5], a_plane[pp][6], a_plane[pp][7],
             a_share_from_ptr + (64 + (line + 1) % 8 * SMEM_STRIDE) * sizeof(float));
      lds128(b_plane[pp][0], b_plane[pp][1], b_plane[pp][2], b_plane[pp][3],
             b_share_from_ptr + (line + 1) % 8 * SMEM_STRIDE * sizeof(float));
      lds128(b_plane[pp][4], b_plane[pp][5], b_plane[pp][6], b_plane[pp][7],
             b_share_from_ptr + (64 + (line + 1) % 8 * SMEM_STRIDE) * sizeof(float));

      if (line == 0 && kk < k - 8)
      {
        for (int i = 0; i < 4; ++i)
        {
          ld_global_to_reg_32(a_reg[i], (const char *)(a_from_ptr) + i * k * sizeof(float));
        }
        for (int i = 0; i < 4; ++i)
        {
          ld_global_to_reg_32(b_reg[i], (const char *)(b_from_ptr) + i * 32 * sizeof(float));
        }
        __syncthreads();
      }

      for (int i = 0; i < 8; ++i)
      {
        for (int j = 0; j < 8; ++j)
        {
          sum[i][j] += a_plane[line % 2][i] * b_plane[line % 2][j];
        }
      }
    }
  }
#if 0
  float *c_to = c + by * BLOCK * n + bx * BLOCK + (tx / 16 * 4) * n + (tx % 16 * 4);
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      c_to[i * n + j] = sum[i][j];
      c_to[i * n + j + 64] = sum[i][j + 4];
      c_to[(i + 64) * n + j] = sum[i + 4][j];
      c_to[(i + 64) * n + j + 64] = sum[i + 4][j + 4];
    }
  }
#else
  uint32_t c_sptr = smem_addr_to_u32(reinterpret_cast<float *>(smem) + tx / 16 * 4 * SMEM_STRIDE_C + tx % 16 * 4);
  float *c_s_to_g_ptr = reinterpret_cast<float *>(smem) + tx / 32 * 8 * SMEM_STRIDE_C + tx % 32;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      __syncthreads();
      for (int p = 0; p < 4; ++p)
      {
        sts128(sum[i * 4 + p][j * 4], sum[i * 4 + p][j * 4 + 1],
               sum[i * 4 + p][j * 4 + 2], sum[i * 4 + p][j * 4 + 3],
               c_sptr + p * SMEM_STRIDE_C * sizeof(float));
      }
      __syncthreads();

      float *c_ptr = c + by * BLOCK * n + bx * BLOCK + (i * 64 + tx / 32 * 8) * n + (j * 64 + tx % 32);
      for (int p = 0; p < 8; ++p)
      {
        stg32(c_s_to_g_ptr[p * SMEM_STRIDE_C], c_ptr + p * n);
        stg32(c_s_to_g_ptr[p * SMEM_STRIDE_C + 32], c_ptr + p * n + 32);
      }
    }
  }
#endif
}

void MY_MMult(cublasContext *handle, int n, int m, int k, float *a, int len_a, float *b, int len_b, float *c, int len_c)
{
  constexpr int BLOCK = 128;
  dim3 block(BLOCK * 2);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm<BLOCK><<<grid, block>>>(n, m, k, a, b, c);
}