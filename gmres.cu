#include "gmres.h"


void cublasCheck(cublasStatus_t stat, const char* function_name)
{
    if(stat!=CUBLAS_STATUS_SUCCESS)
        printf("%s failed\n",function_name);
}
int cusparseStatusCheck(cusparseStatus_t Stat)
{
  if(Stat==CUSPARSE_STATUS_SUCCESS)
    return 0;
  else if(Stat==CUSPARSE_STATUS_NOT_INITIALIZED)
    return -2;
  else if(Stat==CUSPARSE_STATUS_ALLOC_FAILED)
    return -3;
  else if(Stat==CUSPARSE_STATUS_INVALID_VALUE)
    return -4;
  else if(Stat==CUSPARSE_STATUS_ARCH_MISMATCH)
    return -5;
  else if(Stat==CUSPARSE_STATUS_EXECUTION_FAILED)
    return -6;
  else if(Stat==CUSPARSE_STATUS_INTERNAL_ERROR)
    return -7;
  else if(Stat==CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
    return -8;
  else if(Stat==CUSPARSE_STATUS_NOT_SUPPORTED)
    return -9;
  else if(Stat==CUSPARSE_STATUS_INSUFFICIENT_RESOURCES)
    return -10;
  return -11;
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


__global__ void precondition(double* A,double* A_copy,double*b,int matrix_dim)
{
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    if(tx>=matrix_dim*matrix_dim)
      return;
    int row=tx%matrix_dim;
    int col=tx/matrix_dim;
    if(row==col)
      b[row]=b[row]/A_copy[IDX2C(row,row,matrix_dim)];
    A[IDX2C(row,col,matrix_dim)]=A[IDX2C(row,col,matrix_dim)]/A_copy[IDX2C(row,row,matrix_dim)];
}

__global__ void getNegScalar(double* destination, double* source)
{
  *destination=*source;
}

// Adapt based on: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
void GMRES(cublasHandle_t handle,cudaStream_t stream_id, double* A, double*b, double* x, double* Q, double* H,const int matrix_dim,const int max_iterations, const double threshold)
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

    //double* A_copy;//may need to be initialized to zero
    //cudaMallocManaged(&A_copy, sizeof(double)*matrix_dim*matrix_dim);
    //cudaMemcpy(A_copy,A,sizeof(double)*matrix_dim*matrix_dim,cudaMemcpyDefault);

    //int threads_per_block=1024;
    //int blocks_per_grid=(matrix_dim*matrix_dim+threads_per_block-1)/threads_per_block;
    //dim3 dimGrid(blocks_per_grid); // one-dimensional grid
    //dim3 dimBlock(threads_per_block);

    //precondition<<<dimGrid,dimBlock,0,stream_id>>>(A,A_copy,b,matrix_dim);

    // printf("A===================\n");
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%.3f,",A[IDX2C(i,j,matrix_dim)]);
    //     }
    //     printf("\n");
    // }

    // printf("b=================\n");
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     printf("%.3f\n",b[i]);
    // }


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
    cudaDeviceSynchronize();
    //CPU
    double error;

    error=r_norm/b_norm;


    double r_norm_reciprocal=0;//r_norm_reciprocal is on the host memory

    r_norm_reciprocal=1/r_norm;//do on the host


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
        cudaDeviceSynchronize();
        //CPU
        rotate_Hessenberg((double*)(H+k*matrix_dim), cs, sn, k);//assume k<=1024
        next_sin_cos(*(H+k*matrix_dim+k),*(H+k*matrix_dim+k+1), (double*)(cs+k), (double*)(sn+k));
        *(H+k*matrix_dim+k)=cs[k]*(*(H+k*matrix_dim+k))+sn[k]*(*(H+k*matrix_dim+k+1));
        *(H+k*matrix_dim+k+1)=0;

        //update the residual vector
        beta_r[k + 1] = -sn[k] * beta_r[k];
        beta_r[k]     = cs[k] * beta_r[k];
        error       = abs(beta_r[k + 1]) / b_norm;

        //printf("error:%f\n",error);

        if(error<=threshold)
            break;
    }

    // printf("Q===================\n");
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%.3f,",Q[IDX2C(i,j,matrix_dim)]);
    //     }
    //     printf("\n");
    // }
    // printf("H===================\n");
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%.3f,",H[IDX2C(i,j,matrix_dim)]);
    //     }
    //     printf("\n");
    // }

    // printf("beta===============\n");
    // for(int i=0;i<=k_record+1;i++)
    // {
    //     printf("%.3f\n",beta_r[i]);
    // }

    printf("%f\n",error);
    double *y;
    cudaMallocManaged(&y, sizeof(double)*(k_record+1));
    back_substituition(H, beta_r,k_record+1, y,matrix_dim);

    // printf("y==============\n");
    // for(int i=0;i<k_record+1;i++)
    // {
    //     printf("%.3f\n",y[i]);
    // }

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
        cudaDeviceSynchronize();
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

    cudaDeviceSynchronize();
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

// A (device) is stored in column-major order, Q (device) is 2D array, Q[i] means Qth column
// k is the total finished column index
// q vector (device) has dimension matrix_dim; h vector (device) has dimension matrix_dim+1
void arnoldi_sparse(cublasHandle_t handle,cusparseHandle_t handle_sparse, int* bsrRowPtrC,int* bsrColIndC,double* bsrValC, int mb, int nb, int nnzb, int blockDim,
          double* Q, double *H, const int k, const int matrix_dim,cudaStream_t stream_id)
{
    cublasStatus_t cudaStat;

    cusparseMatDescr_t descrC;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

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
    
  
    //q = A*Q(:,k);
    alpha=1;
    beta=0;
    cusparseDbsrmv(handle_sparse, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE , mb, nb, nnzb, &alpha,
       descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, (double*) (Q+k*matrix_dim), &beta, q);
    // cudaStat=cublasDgemv(handle, CUBLAS_OP_N,
    //                        matrix_dim, matrix_dim,
    //                        &alpha,
    //                        A, matrix_dim,
    //                        (double*) (Q+k*matrix_dim), 1,
    //                        &beta,
    //                        q, 1);
    // cublasCheck(cudaStat,"cublasDgemv");   

    for(int i=0;i<k+1;i++)
    {
        // h(i) = q' * Q(:, i);
        cudaStat=cublasDdot (handle, matrix_dim,
                           q, 1,
                           (double*)(Q+i*matrix_dim), 1,
                           (double*)(h+i));
                        
        cublasCheck(cudaStat,"cublasDdot");
        cudaDeviceSynchronize();
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

    cudaDeviceSynchronize();
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
    
    cusparseDestroyMatDescr(descrC);
    cudaFree(q_norm);
    cudaFree(q);
    cudaFree(h);

}

// Adapt based on: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
void GMRES_sparse(cublasHandle_t handle,cusparseHandle_t handle_sparse,cudaStream_t stream_id, 
                double* csrValA, int* csrRowPtrA, int* csrColIndA, double*b, double* x, double* Q, double* H,const int matrix_dim,const int max_iterations, const double threshold)
{
    cublasStatus_t cudaStat;
    cudaStat=cublasSetStream(handle, stream_id);
    cusparseSetStream(handle_sparse,stream_id);

    

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

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseMatDescr_t descrC;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
    
     // from CSR to BSR
    int nnzb;
    int blockDim=2;
    int m=matrix_dim;
    int n=matrix_dim;
    int mb = (m + blockDim-1)/blockDim;
    int nb = (n + blockDim-1)/blockDim;
    //printf("nb=%d\n",nb);

    int* bsrRowPtrC;
    int* bsrColIndC;
    double* bsrValC;
    cudaMallocManaged(&bsrRowPtrC, sizeof(int) *(mb+1));
    cusparseXcsr2bsrNnz(handle_sparse, CUSPARSE_DIRECTION_ROW, m, n,
        descrA , csrRowPtrA, csrColIndA, blockDim,
        descrC , bsrRowPtrC, &nnzb);
    //printf("nnzb=%d\n",nnzb);
    
    cudaMallocManaged(&bsrColIndC, sizeof(int)*nnzb);
    cudaMallocManaged(&bsrValC, sizeof(double)*(blockDim*blockDim)*nnzb);
    cusparseDcsr2bsr(handle_sparse, CUSPARSE_DIRECTION_ROW, m, n,
        descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
        descrC, bsrValC, bsrRowPtrC, bsrColIndC);
    alpha=-1;
    beta=1;
    cusparseDbsrmv(handle_sparse, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE , mb, nb, nnzb, &alpha,
       descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, x, &beta, r);

    //r=b-A*x;
    // cudaStat=cublasDgemv(handle, CUBLAS_OP_N,
    //                        matrix_dim, matrix_dim,
    //                        &alpha,
    //                        A, matrix_dim,
    //                        x, 1,
    //                        &beta,
    //                        r, 1);

    // cublasCheck(cudaStat,"cublasDgemv");

    //r_norm = norm(r);
    double r_norm=0;//r_norm is on the host memory
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            r, 1, &r_norm);//probably blocking to make sure the correct r_norm is on the host
    cublasCheck(cudaStat,"cublasDnrm2");

    double b_norm=0;
    cudaStat = cublasDnrm2( handle, matrix_dim,
                            r, 1, &b_norm);//probably blocking to make sure the correct r_norm is on the host
    cublasCheck(cudaStat,"cublasDnrm2");
    cudaDeviceSynchronize();
    //CPU
    double error;

    error=r_norm/b_norm;


    double r_norm_reciprocal=0;//r_norm_reciprocal is on the host memory

    r_norm_reciprocal=1/r_norm;//do on the host


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
        arnoldi_sparse(handle,handle_sparse,  bsrRowPtrC,bsrColIndC,bsrValC,  mb,  nb, nnzb, blockDim,
          Q,H, k, matrix_dim,stream_id);
        cudaDeviceSynchronize();
        //CPU
        rotate_Hessenberg((double*)(H+k*matrix_dim), cs, sn, k);//assume k<=1024
        next_sin_cos(*(H+k*matrix_dim+k),*(H+k*matrix_dim+k+1), (double*)(cs+k), (double*)(sn+k));
        *(H+k*matrix_dim+k)=cs[k]*(*(H+k*matrix_dim+k))+sn[k]*(*(H+k*matrix_dim+k+1));
        *(H+k*matrix_dim+k+1)=0;

        //update the residual vector
        beta_r[k + 1] = -sn[k] * beta_r[k];
        beta_r[k]     = cs[k] * beta_r[k];
        error       = abs(beta_r[k + 1]) / b_norm;

        //printf("error:%f\n",error);

        if(error<=threshold)
            break;
    }

    printf("%f\n",error);
    double *y;
    cudaMallocManaged(&y, sizeof(double)*(k_record+1));
    back_substituition(H, beta_r,k_record+1, y,matrix_dim);

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


    cudaFree(r);
    cudaFree(beta_r);
    cudaFree(sn);
    cudaFree(cs);
    cudaFree(e1);
    cudaFree(y);
}

