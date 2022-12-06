#include "gmres.h"
#include <cuda.h>
void cublasCheck(cublasStatus_t stat, const char* function_name)
{
    printf("%s\n",function_name);
    if(stat!=CUBLAS_STATUS_SUCCESS)
        printf("%s failed\n",function_name);
}

__global__ void scalar_reciprocal(double* value)
{
    printf("scalar reciprocal\n");
    if(*value!=0)
    *value=1/(*value);
}


__global__ void element_append_vector(double* h, int k, double value)
{
    printf("element_append_vector\n");
    h[k+1]=value;
}

void GMRES(cublasHandle_t handle,const double* A, double*b, double* x, double* Q, double* H,const int matrix_dim,cudaStream_t stream_id)
{
    cublasStatus_t cudaStat;
    cudaStat=cublasSetStream(handle, stream_id);

    double alpha=-1;
    double beta=1;

    double *r;
    cudaMallocManaged(&r, sizeof(double) * matrix_dim);
    cudaMemcpy(r,b,sizeof(double) *matrix_dim,cudaMemcpyDefault);

    //r=b-A*x;
    cudaStat=cublasDgemv(handle, CUBLAS_OP_N,
                           matrix_dim, matrix_dim,
                           &alpha,
                           A, matrix_dim,
                           x, 1,
                           &beta,
                           r, 1);

    cublasCheck(cudaStat,"cublasDgemv");

    //r_norm = norm(r);
    double r_norm=0;
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            r, 1, &r_norm);
    cublasCheck(cudaStat,"cublasDnrm2");

    double r_norm_reciprocal=0;
    if(r_norm!=0)
    {
        r_norm_reciprocal=1/r_norm;
    }

    //r = r / r_norm;
    cudaStat = cublasDscal(handle, matrix_dim,
                            &r_norm_reciprocal,
                            r, 1);
    cublasCheck(cudaStat,"cublasDnrm2");
    

    cudaMemcpy(Q,r,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    cudaDeviceSynchronize();

    int k=0;
    arnoldi(handle, A,  Q, H, k, matrix_dim,stream_id);
    cudaDeviceSynchronize();


    cudaFree(r);

}

// A (device) is stored in column-major order, Q (device) is 2D array, Q[i] means Qth column
// k is the total finished column index
// q vector (device) has dimension matrix_dim; h vector (device) has dimension matrix_dim+1
void arnoldi(cublasHandle_t handle,const double* A, double* Q, double *H, const int k, const int matrix_dim,cudaStream_t stream_id)
{
    cublasStatus_t cudaStat;
    cudaStat=cublasSetStream(handle, stream_id);
    cublasCheck(cudaStat,"cublasSetStream");


    double alpha=1;
    double beta=0;

    double*q;
    cudaMallocManaged(&q, sizeof(double) * matrix_dim);
    for(int i=0;i<matrix_dim;i++)
    {
        q[i]=0;
    }
    double* h;
    cudaMallocManaged(&h, sizeof(double) * matrix_dim);
    for(int i=0;i<matrix_dim;i++)
    {
        h[i]=0;
    }


    double* q_norm;
    cudaMallocManaged(&q_norm, sizeof(double) * 1);
    
    
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%.3f,",Q[IDX2C(i,j,matrix_dim)]);
        }
        printf("\n");
    }
    cudaStat=cublasDgemv(handle, CUBLAS_OP_N,
                           matrix_dim, matrix_dim,
                           &alpha,
                           A, matrix_dim,
                           (double*) (Q+k*matrix_dim), 1,
                           &beta,
                           q, 1);
    cudaDeviceSynchronize();
    for(int i=0;i<matrix_dim;i++)
    {
        printf("%.3f\n",q[i]);
    }
    

    cublasCheck(cudaStat,"cublasDgemv");

    for(int i=0;i<k+1;i++)
    {
        // h(i) = q' * Q(:, i);
        cudaStat=cublasDdot (handle, matrix_dim,
                           q, 1,
                           (double*)(Q+i*matrix_dim), 1,
                           (double*)(h+i));
                        
        cublasCheck(cudaStat,"cublasDdot");
        cudaDeviceSynchronize();
        printf("%.3f\n",*(h+i));
        //q = q - h(i) * Q(:, i);
        alpha=-(*(h+i));
        cudaStat = cublasDaxpy(handle, matrix_dim,
                           &alpha,
                           (double*)(Q+i*matrix_dim), 1,
                           q, 1);
        cublasCheck(cudaStat,"cublasDaxpy");
    }
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            q, 1, q_norm);
    cublasCheck(cudaStat,"cublasDnrm2");

    element_append_vector<<<1,1,0,stream_id>>> (h, k, *q_norm);

    //should be in the same stream as the cublas
    // q_norm=1/q_norm;
    scalar_reciprocal<<<1,1,0,stream_id>>> (q_norm);
    //cudaDeviceSynchronize();
    //printf("q_norm:%.3f\n",*q_norm);

    // q = q / h(k + 1);
    cudaStat = cublasDscal(handle, matrix_dim,
                            q_norm,
                            q, 1);
    cublasCheck(cudaStat,"cublasDnrm2");
    

    cudaDeviceSynchronize();

    cudaMemcpy((double*)(Q+(k+1)*matrix_dim),q,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    cudaMemcpy((double*)(H+(k+1)*matrix_dim),h,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    
    cudaFree(q_norm);
    cudaFree(q);
    cudaFree(h);

}

