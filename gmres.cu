#include "gmres.h"
#include <cuda.h>
void cublasCheck(cublasStatus_t stat, const char* function_name)
{
    //printf("%s\n",function_name);
    if(stat!=CUBLAS_STATUS_SUCCESS)
        printf("%s failed\n",function_name);
}



__global__ void element_append_vector(double* h, int k, double value)
{
    //printf("element_append_vector\n");
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
    double r_norm=0;//r_norm is on the host memory
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            r, 1, &r_norm);//probably blocking to make sure the correct r_norm is on the host
    cublasCheck(cudaStat,"cublasDnrm2");

    double r_norm_reciprocal=0;//r_norm_reciprocal is on the host memory
    if(r_norm!=0)
    {
        r_norm_reciprocal=1/r_norm;//do on the host
    }

    //r = r / r_norm;
    cudaStat = cublasDscal(handle, matrix_dim,
                            &r_norm_reciprocal,
                            r, 1);
    cublasCheck(cudaStat,"cublasDscal");

    cudaMemcpy(Q,r,sizeof(double) *matrix_dim,cudaMemcpyDefault);

    int k=0;
    arnoldi(handle, A,  Q, H, k, matrix_dim,stream_id);

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
    
    
    // debugging printf
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%.3f,",Q[IDX2C(i,j,matrix_dim)]);
    //     }
    //     printf("\n");
    // }

    //q = A*Q(:,k);
    cudaStat=cublasDgemv(handle, CUBLAS_OP_N,
                           matrix_dim, matrix_dim,
                           &alpha,
                           A, matrix_dim,
                           (double*) (Q+k*matrix_dim), 1,
                           &beta,
                           q, 1);
    cublasCheck(cudaStat,"cublasDgemv");   

    for(int i=0;i<k+1;i++)
    {
        // h(i) = q' * Q(:, i);
        cudaStat=cublasDdot (handle, matrix_dim,
                           q, 1,
                           (double*)(Q+i*matrix_dim), 1,
                           (double*)(h+i));
                        
        cublasCheck(cudaStat,"cublasDdot");

        alpha=-*(h+i);//alpha should be the const in cublasDaxpy so the cudaDeviceSynchronize must be added
        //q = q - h(i) * Q(:, i);
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


    //seems there is a copy from device memory to host memory and is blocking
    double q_norm_reciprocal=1/(*q_norm);
    //printf("q_norm_reciprocal:%.3f\n",q_norm_reciprocal);

    // q = q / h(k + 1);
    cudaStat=cublasSetStream(handle, stream_id);
    cublasCheck(cudaStat,"cublasSetStream");

    cudaStat = cublasDscal(handle, matrix_dim,
                            &q_norm_reciprocal,
                            q, 1);
    cublasCheck(cudaStat,"cublasDnrm2");
    

    cudaMemcpy((double*)(Q+(k+1)*matrix_dim),q,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    cudaMemcpy((double*)(H+(k+1)*matrix_dim),h,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    
   
    cudaFree(q_norm);
    cudaFree(q);
    cudaFree(h);

}

