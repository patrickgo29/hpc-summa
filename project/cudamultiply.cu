#include "cudamultiply.h"

__global__ void kernelFunc(int m, int n, int k, float* ad, float* bd, float* cd) {
    double v = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ind;
    for (ind = 0; ind < k; ++ind)
    {
       v += ad[row+ind*m]*bd[ind+col*k];
    }

   cd[row+m*col] += Ctemp + cd[row+m*col];
   __syncthreads();
}

void matrix_multiply_cuda(int m, int n, int k,
	      const double* A, int lda, const double* B, int ldb,
	      double* C, int ldc) {
		  
	assert (A || m <= 0 || k <= 0); assert (lda >= m);
	assert (B || k <= 0 || n <= 0); assert (ldb >= k);
	assert (C || m <= 0 || n <= 0); assert (ldc >= m);	  
	
    float* ad;
    float* bd;
    float* cd;
    
    cudaMalloc((void**)&ad, m * k * sizeof(float));
    cudaMalloc((void**)&bd, k * n * sizeof(float));
    cudaMalloc((void**)&cd, m * n * sizeof(float));
    
    cudaMemcpy(ad, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cd, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

	// What dimension?
    dim3 block(?, ?);           
    dim3 grid(?, ?);
    
    kernelFunc<<<grid, block>>>(m,n,k,ad, bd, cd);

    cudaMemcpy(c, cd, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}