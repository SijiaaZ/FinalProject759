#include "gmres.h"
#include <cuda.h>

void cublasCheck(cublasStatus_t stat, const char* function_name)
{
    //printf("%s\n",function_name);
    if(stat!=CUBLAS_STATUS_SUCCESS)
        printf("%s failed\n",function_name);
}

//k is zero based, the target column index, k+1
void rotate_Hessenberg(double* h, double* cs, double* sn, const int k)
{
    double temp=0;

    for(int i=0;i<k;i++)
    {   
        temp   =  cs[i] * h[i] + sn[i] * h[i + 1];
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i]   = temp;
    }
    
}

void next_sin_cos(double v1,double v2, double* cs, double* sn)
{
    double t = sqrt(v1*v1 + v2*v2);

    *cs = v1 / t;
    *sn = v2 /t;

}

//A is a square matrix, column based
// change the reading sequence
void back_substituition(const double* A, double *b,int matrix_dim, double *x, int lda)
{
    for(int r=matrix_dim-1;r>=0;r--)//row to be solved
    {
        for(int c=matrix_dim-1;c>r;c--)
        {
            b[r]-=x[c]*A[IDX2C(r,c,lda)];
        }
        x[r]=b[r]/A[IDX2C(r,r,lda)];
    }
}
// __global__ void element_append_vector(double* h, int k, double value)
// {
//     //printf("element_append_vector\n");
//     h[k+1]=value;
// }
// Adapt based on: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
void GMRES(cublasHandle_t handle,cudaStream_t stream_id,const double* A, double*b, double* x, double* Q, double* H,const int matrix_dim,const int max_iterations, const double threshold)
{
    cublasStatus_t cudaStat;
    cudaStat=cublasSetStream(handle, stream_id);

    double alpha=-1;
    double beta=1;

    double *r;
    cudaMallocManaged(&r, sizeof(double) * matrix_dim);
    cudaMemcpy(r,b,sizeof(double) *matrix_dim,cudaMemcpyDefault);

    double *cs;//may need to be initialized to zero
    cudaMallocManaged(&cs, sizeof(double) * max_iterations);

    double *sn;//may need to be initialized to zero
    cudaMallocManaged(&sn, sizeof(double) * max_iterations);

    double* e1;//may need to be initialized to zero
    cudaMallocManaged(&e1, sizeof(double) * (max_iterations+1));
    e1[0]=1;

    double* beta_r;//may need to be initialized to zero
    cudaMallocManaged(&beta_r, sizeof(double) * (max_iterations+1));
    beta_r[0]=1;

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

    double b_norm=0;
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            r, 1, &b_norm);//probably blocking to make sure the correct r_norm is on the host
    cublasCheck(cudaStat,"cublasDnrm2");

    double error;
    if(b_norm!=0)
    {
        error=r_norm/b_norm;
    }

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
    //beta = r_norm * e1;
    beta_r[0] = r_norm * beta_r[0];
    int k_record=0;
    for(int k=0;k<max_iterations;k++)
    {
        k_record=k;
        arnoldi(handle, A,  Q, H, k, matrix_dim,stream_id);
        rotate_Hessenberg((double*)(H+k*matrix_dim), cs, sn, k);//assume k<=1024
        cudaStreamSynchronize(stream_id);
        next_sin_cos(*(H+k*matrix_dim+k),*(H+k*matrix_dim+k+1), (double*)(cs+k), (double*)(sn+k));
        *(H+k*matrix_dim+k)=cs[k]*(*(H+k*matrix_dim+k))+sn[k]*(*(H+k*matrix_dim+k+1));
        *(H+k*matrix_dim+k+1)=0;

        //update the residual vector
        beta_r[k + 1] = -sn[k] * beta_r[k];
        beta_r[k]     = cs[k] * beta_r[k];
        error       = abs(beta_r[k + 1]) / b_norm;

        if(error<=threshold)
            break;
    }

    printf("Q===================\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%.3f,",Q[IDX2C(i,j,matrix_dim)]);
        }
        printf("\n");
    }
    printf("H===================\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%.3f,",H[IDX2C(i,j,matrix_dim)]);
        }
        printf("\n");
    }

    printf("beta===============\n");
    for(int i=0;i<=k_record+1;i++)
    {
        printf("%.3f\n",beta_r[i]);
    }


    double *y;
    cudaMallocManaged(&y, sizeof(double)*(k_record+1));
    back_substituition(H, beta_r,k_record+1, y,matrix_dim);

    printf("y==============\n");
    for(int i=0;i<k_record+1;i++)
    {
        printf("%.3f\n",y[i]);
    }

    //x = x + Q(:, 1:k) * y;
    alpha=1;
    beta=1;
    cudaStat = cublasDgemv(handle, CUBLAS_OP_N,
                           matrix_dim, k_record+1,
                           &alpha,
                           Q, matrix_dim,
                           y, 1,
                           &beta,
                           x, 1);
    cublasCheck(cudaStat,"cublasDgemv");
    cudaDeviceSynchronize();

    printf("x===================\n");
    for(int i=0;i<matrix_dim;i++) 
    {
        printf("%.3f\n",x[i]);

    }

    cudaFree(r);
    cudaFree(beta_r);
    cudaFree(sn);
    cudaFree(cs);
    cudaFree(e1);
    cudaFree(y);
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
    cudaMemcpy((double*)(h+k+1),q_norm,sizeof(double)*1,cudaMemcpyDefault);


    //seems there is a copy from device memory to host memory and is blocking
    double q_norm_reciprocal=1/(*q_norm);
    //printf("q_norm_reciprocal:%.3f\n",q_norm_reciprocal);

    // q = q / h(k + 1);

    cudaStat = cublasDscal(handle, matrix_dim,
                            &q_norm_reciprocal,
                            q, 1);
    cublasCheck(cudaStat,"cublasDnrm2");
    

    cudaMemcpy((double*)(Q+(k+1)*matrix_dim),q,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    cudaMemcpy((double*)(H+k*matrix_dim),h,sizeof(double) *matrix_dim,cudaMemcpyDefault);
    
   
    cudaFree(q_norm);
    cudaFree(q);
    cudaFree(h);

}

