#ifndef gmres_h
#define gmres_h
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda.h>
#include <cusparse.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
void GMRES(cublasHandle_t handle,cudaStream_t stream_id, double* A, double*b, double* x, double* Q, double* H,const int matrix_dim,const int max_iterations, const double threshold);
void arnoldi(cublasHandle_t handle,const double* A, double* Q, double *H, const int k, const int matrix_dim,cudaStream_t stream_id);
void GMRES_sparse(cublasHandle_t handle,cusparseHandle_t handle_sparse,cudaStream_t stream_id, 
                double* csrValA, int* csrRowPtrA, int* csrColIndA, double*b, double* x, double* Q, double* H,const int matrix_dim,const int max_iterations, const double threshold);
void arnoldi_sparse(cublasHandle_t handle,cusparseHandle_t handle_sparse, int* bsrRowPtrC,int* bsrColIndC,double* bsrValC, int mb, int nb, int nnzb, int blockDim,
          double* Q, double *H, const int k, const int matrix_dim,cudaStream_t stream_id);
#endif