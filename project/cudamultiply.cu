#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#define THREADS_PER_BLOCK 4

__global__ void kernelFunc(int m, int n, int k, double* ad, double* bd, double* cd, int lda, int ldb, int ldc) {
	double v = 0.0;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int ind;
	int aIndex;
	int bIndex;
	int cIndex;
	for (ind = 0; ind < k; ++ind)
	{
		aIndex = row+ind*lda;
		bIndex = ind+col*ldb;
		if (aIndex < m*k && bIndex < k*n) {
			v += ad[aIndex]*bd[bIndex];
		}
	}

	cIndex = row+col*ldc;
	if (cIndex < m*n) {
		cd[cIndex] += v;
	}
	__syncthreads();
}

extern "C" void mat_multiply_cuda(int m, int n, int k,
	      const double* A, int lda, const double* B, int ldb,
	      double* C, int ldc) {
		  
	assert (A || m <= 0 || k <= 0); assert (lda >= m);
	assert (B || k <= 0 || n <= 0); assert (ldb >= k);
	assert (C || m <= 0 || n <= 0); assert (ldc >= m);	  
	
	double* ad;
	double* bd;
	double* cd;
    
	cudaMalloc((void**)&ad, m * k * sizeof(double));
	cudaMalloc((void**)&bd, k * n * sizeof(double));
	cudaMalloc((void**)&cd, m * n * sizeof(double));
	
	cudaMemcpy(ad, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(bd, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cd, C, m * n * sizeof(double), cudaMemcpyHostToDevice);
	
	int size = THREADS_PER_BLOCK;
	dim3 block(size, size);           
	dim3 grid((n+size-1)/size, (m+size-1)/size);
	
	kernelFunc<<<grid, block>>>(m,n,k,ad, bd, cd, lda, ldb, ldc);
	
	cudaMemcpy(C, cd, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
}