#ifndef lu_factorization_h
#define lu_factorization_h
#include <cuda.h>
__global__ void gaussian_elemination_one_entry(int pivot_row, double* conductances,double* L, double* U, int matrix_dim);
void LU_factorization_GPU( double* conductances, double* L, double* U, int matrix_dim);
#endif
