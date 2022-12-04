#include "lu_factorization.h"

__global__ void gaussian_elemination_one_entry(int pivot_row,double* conductances,double* L, double* U, int matrix_dim)
{
    extern __shared__ double conductances_shared[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int entry=tx*matrix_dim+ty;

    conductances_shared[entry]=conductances[entry];

    __syncthreads();// all the necessary data is stored in shared memory

    // do repetitve computation for entries in L but eliminate the need for __syncthreads() again
    double l_tx_ty=conductances_shared[tx*matrix_dim+pivot_row]/conductances_shared[pivot_row*matrix_dim+pivot_row];
    
    if(tx>pivot_row)
        conductances_shared[entry]-=L[tx*matrix_dim+pivot_row];

    if(tx>pivot_row&&pivot_row==ty)//store it in L matrix
    {
        L[tx*matrix_dim+pivot_row]=l_tx_ty;
    }

    if(tx<=pivot_row&&pivot_row==ty)//store it in U matrix
    {
        U[tx*matrix_dim+pivot_row]=conductances_shared[entry];
    }

    conductances[entry]=conductances_shared[entry];

}

__host__  void LU_factorization_GPU( double* conductances, double* L, double* U, int matrix_dim)
{

    int blocks_per_grid=1;
    dim3 dimGrid(blocks_per_grid); // one-dimensional grid
    dim3 dimBlock(matrix_dim,matrix_dim);
    int shared_mem_size=(matrix_dim*matrix_dim)*sizeof(double);

    for(int i=0;i<matrix_dim-1;i++)
    {
        gaussian_elemination_one_entry<<<dimGrid, dimBlock,shared_mem_size>>>(i,conductances,L,U,matrix_dim);
    }
    cudaDeviceSynchronize();
}


