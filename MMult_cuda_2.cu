#include <cublas.h>


__global__ void sgemm(int m, int n, int k, float* A, int len_a, float* B, int len_b, float* C, int len_c){
    int _n = blockIdx.x * blockDim.x + threadIdx.x;
    int _m = blockIdx.y * blockDim.y + threadIdx.y;
    if(_m < m && _n < n){
        float sum = 0;
        for(int i = 0 ; i < k ; ++i){
            sum += A[_m * k + i] * B[n * i + _n];
        }
        C[_m * n + _n] = sum;
    }
}

void MY_MMult(cublasContext* handle, int m, int n, int k, float* A, int len_a, float* B, int len_b, float* C, int len_c){
    constexpr int BLOCK=16;
    dim3 block (BLOCK, BLOCK);
    dim3 grid ((m + BLOCK - 1)/ BLOCK, (n + BLOCK - 1)/ BLOCK);
    sgemm<<<grid, block>>>(m, n , k , A, len_a, B, len_b, C, len_c);
}