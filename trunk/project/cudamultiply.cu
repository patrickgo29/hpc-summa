#include <assert.h>
#define THREADS_PER_BLOCK 4

__global__ void kernelFunc(int m, int n, int k, float* ad, float* bd, float* cd, int lda, int ldb, int ldc) {
    double v = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
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

	cIndex = row+m*col;
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
	
    float* ad;
    float* bd;
    float* cd;
    
    cudaMalloc((void**)&ad, m * k * sizeof(float));
    cudaMalloc((void**)&bd, k * n * sizeof(float));
    cudaMalloc((void**)&cd, m * n * sizeof(float));
    
    cudaMemcpy(ad, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cd, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

	int size = THREADS_PER_BLOCK;
    dim3 block(size, size);           
    dim3 grid((n+size-1)/size, (m+size-1)/size);
    
    kernelFunc<<<grid, block>>>(m,n,k,ad, bd, cd, lda, ldb, ldc);

    cudaMemcpy(C, cd, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}