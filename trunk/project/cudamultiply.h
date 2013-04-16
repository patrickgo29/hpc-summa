void matrix_multiply_cuda(int m, int n, int k,
						  const double* A, int lda, const double* B, int ldb,
						  double* C, int ldc);