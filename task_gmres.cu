#include "gmres.h"
#include <iostream>

int main(int argc, char *argv[]) {

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    int matrix_dim=4;

    double* A_h=new double[matrix_dim * matrix_dim]{1,-0.5,0,0,0,1,0,0,0,0,1,0,0.5,0,0,1};
    double* A;
    cudaMallocManaged(&A, sizeof(double) * matrix_dim * matrix_dim);
    cudaMemcpy(A, A_h, sizeof(double) * matrix_dim * matrix_dim, cudaMemcpyDefault);

    double* b_h=new double[matrix_dim]{0.35,0.25,0.2,0.1};
    double* b;
    cudaMallocManaged(&b, sizeof(double) * matrix_dim );
    cudaMemcpy(b, b_h, sizeof(double) * matrix_dim , cudaMemcpyDefault);

    double* Q;
    cudaMallocManaged(&Q, sizeof(double) * matrix_dim * matrix_dim);
    for(int i=0;i<matrix_dim*matrix_dim;i++)
    {
        Q[i]=0;
    }
    std::cout<<Q<<std::endl;
    std::cout<<(double*) (Q+matrix_dim)<<std::endl;
    std::cout<<(double*) (Q+matrix_dim)-Q<<std::endl;
    std::cout<<&Q[matrix_dim]<<std::endl;
    std::cout<<&Q[matrix_dim]-Q<<std::endl;

    double* H;
    cudaMallocManaged(&H, sizeof(double) * matrix_dim * matrix_dim);
    for(int i=0;i<matrix_dim*matrix_dim;i++)
    {
        H[i]=0;
    }

    double *x;
    cudaMallocManaged(&x, sizeof(double) * matrix_dim );


    GMRES(handle,A, b, x, Q, H, matrix_dim,stream1);

    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%.3f,",A[IDX2C(i,j,matrix_dim)]);
        }
        printf("\n");
    }
    printf("===================\n");

    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%.3f,",Q[IDX2C(i,j,matrix_dim)]);
        }
        printf("\n");
    }


    
    
    cudaFree(A);
    cudaFree(Q);
    cudaFree(H);
    cublasDestroy(handle);
    cudaStreamDestroy(stream1);

    delete []A_h;
    delete []b_h;

    return 0;
}