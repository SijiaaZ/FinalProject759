#include "gmres.h"
#include "matrix_helper.h"
#include "linear_solver.h"
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <cusparse.h>
#include <cuda_runtime.h>
#include "parse.h"


int main(int argc, char *argv[]) {
    char* filename=argv[1];
    int elementNum=0;
    Element* elementList=parseNetlist(filename, elementNum);
    int augmented_matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementNum);

    int matrix_dim=augmented_matrix_dim-1;
    double* conductance=new double[augmented_matrix_dim*augmented_matrix_dim]();
    double* currents=new double[augmented_matrix_dim]();
    double* conductance_definite=new double[matrix_dim*matrix_dim]();
    double* currents_definite=new double[matrix_dim]();
    
    // need to change to the column majored
    elementList_to_augmented_Matrix(elementList, elementNum, conductance, currents, augmented_matrix_dim);
    augmented_Matrix_to_definite_matrix( elementNum,  conductance,  currents,  conductance_definite, currents_definite,  augmented_matrix_dim);
    delete[] conductance;
    delete[] currents;
    delete []elementList;

    FILE * fp;
    fp = fopen ("rand_matrix.out", "w");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            fprintf(fp,"%f,",conductance_definite[i*matrix_dim+j]);
        }
        fprintf(fp,"\n");
    }

    for(int i=0;i<matrix_dim;i++)
    {
        fprintf(fp,"%f\n",currents_definite[i]);
    }
    fclose(fp);

    //dense to CSR
    double* csrValA_h=new double[matrix_dim * matrix_dim];
    int* csrRowPtrA_h=new int[(matrix_dim+1)];
    int* csrColIndA_h=new int[matrix_dim * matrix_dim];
    int nonzeroNums=Dense_to_row_major_CSR(matrix_dim,conductance_definite,csrValA_h, csrRowPtrA_h,csrColIndA_h);
    
    // GMRES
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cusparseHandle_t handle_sparse;
    cusparseCreate(&handle_sparse);

    // cuda memory allocation
    double* csrValA;
    int* csrRowPtrA;
    int* csrColIndA;
    cudaMallocManaged(&csrValA, sizeof(double) * matrix_dim * matrix_dim);
    cudaMallocManaged(&csrRowPtrA, sizeof(int) * (matrix_dim+1));
    cudaMallocManaged(&csrColIndA, sizeof(int) * matrix_dim * matrix_dim);
    
    cudaMemcpy(csrValA, csrValA_h, sizeof(double) * matrix_dim * matrix_dim, cudaMemcpyDefault);
    cudaMemcpy(csrRowPtrA, csrRowPtrA_h, sizeof(int) * (matrix_dim+1), cudaMemcpyDefault);
    cudaMemcpy(csrColIndA, csrColIndA_h, sizeof(int) * matrix_dim * matrix_dim, cudaMemcpyDefault);

    double* b_h=new double[matrix_dim];
    for(int i=0;i<matrix_dim;i++)
    {
        b_h[i]=currents_definite[i];
    }

    double* b;
    cudaMallocManaged(&b, sizeof(double) * matrix_dim );
    cudaMemcpy(b, b_h, sizeof(double) * matrix_dim , cudaMemcpyDefault);

    double* Q;
    cudaMallocManaged(&Q, sizeof(double) * matrix_dim * matrix_dim);
    for(int i=0;i<matrix_dim*matrix_dim;i++)
    {
        Q[i]=0;
    }

    double* H;
    cudaMallocManaged(&H, sizeof(double) * matrix_dim * matrix_dim);
    for(int i=0;i<matrix_dim*matrix_dim;i++)
    {
        H[i]=0;
    }

    double *x;
    cudaMallocManaged(&x, sizeof(double) * matrix_dim );
    //cuda memory allocation finishes

    auto begin = std::chrono::high_resolution_clock::now();
    GMRES_sparse(handle,handle_sparse,stream1,
          csrValA, csrRowPtrA, csrColIndA, b, x, Q,  H,matrix_dim,20, 0.001);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin);
    printf("%f\n", duration_sec.count());

    FILE * fp_Result_3;
    fp_Result_3 = fopen ("nodal_voltages.out", "w");
    for(int i=0;i<matrix_dim;i++)
    {
        fprintf(fp_Result_3,"%.3f\n",x[i]);
    }
    fclose(fp_Result_3);

    // //gaussian elimination
    // gaussian_elimination(conductance_definite, currents_definite, matrix_dim);

    // double* voltages=new double[matrix_dim];
    // back_substituition(conductance_definite, currents_definite, voltages,matrix_dim);

    // FILE * fp_Result_1;
    // fp_Result_1 = fopen ("rand_matrix_result_Gaussian_CPU.out", "w");
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     fprintf(fp_Result_1,"%.3f\n",voltages[i]);
    // }
    // fclose(fp_Result_1);

    // delete []voltages;
    // gaussian elimination finishes

    cudaFree(Q);
    cudaFree(H);
    cudaFree(b);
    cudaFree(x);
    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(csrValA);
    cublasDestroy(handle);
    cusparseDestroy(handle_sparse);
    cudaStreamDestroy(stream1);

    delete []b_h;
    delete []csrValA_h;
    delete []csrColIndA_h;
    delete []csrRowPtrA_h;
    delete []conductance_definite;
    delete []currents_definite;
    

    return EXIT_SUCCESS;
}
// int main(int argc, char *argv[]) {
//     int matrix_dim=6;
//     double* A_h=new double[matrix_dim * matrix_dim]{2,-0.5,0,-0.8,0,0,-0.5,2,0,0,0,0,0,0,4,0,0,0,-0.8,0,0,4,0,0,0,0,0,0,4,0,0,0,0,0,0,1};
//     double* A;
//     cudaMallocManaged(&A,sizeof(double)*matrix_dim*matrix_dim);
//     cudaMemcpy(A, A_h, sizeof(double) * matrix_dim * matrix_dim, cudaMemcpyDefault);
//     double* b_h=new double[matrix_dim]{1,2,2,0,1,1};
//     double* b;
//     cudaMallocManaged(&b,sizeof(double)*matrix_dim);
//     cudaMemcpy(b, b_h, sizeof(double) * matrix_dim, cudaMemcpyDefault);

//     double* csrValA_h=new double[matrix_dim * matrix_dim];
//     int* csrRowPtrA_h=new int[(matrix_dim+1)];
//     int* csrColIndA_h=new int[matrix_dim * matrix_dim];
//     int nonzeroNums=Dense_to_row_major_CSR(matrix_dim,A_h,csrValA_h, csrRowPtrA_h,csrColIndA_h);
//     for(int i=0;i<nonzeroNums;i++)
//     {
//       printf("%f,%d\n",csrValA_h[i],csrColIndA_h[i]);
//     }
//     for(int i=0;i<(matrix_dim+1);i++)
//     {
//       printf("%d\n",csrRowPtrA_h[i]);
//     }

//     double* csrValA;
//     int* csrRowPtrA;
//     int* csrColIndA;
//     cudaMallocManaged(&csrValA, sizeof(double) * matrix_dim * matrix_dim);
//     cudaMallocManaged(&csrRowPtrA, sizeof(int) * (matrix_dim+1));
//     cudaMallocManaged(&csrColIndA, sizeof(int) * matrix_dim * matrix_dim);
    
//     cudaMemcpy(csrValA, csrValA_h, sizeof(double) * matrix_dim * matrix_dim, cudaMemcpyDefault);
//     cudaMemcpy(csrRowPtrA, csrRowPtrA_h, sizeof(int) * (matrix_dim+1), cudaMemcpyDefault);
//     cudaMemcpy(csrColIndA, csrColIndA_h, sizeof(int) * matrix_dim * matrix_dim, cudaMemcpyDefault);


//     //cusparse library begins
//     cusparseHandle_t handle;
//     cusparseStatus_t Stat;
//     Stat= cusparseCreate(&handle);
    

//     cusparseMatDescr_t descrA;
//     Stat=cusparseCreateMatDescr(&descrA);
//     cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
//     cusparseMatDescr_t descrC;
//     Stat=cusparseCreateMatDescr(&descrC);
//     cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
    

//     // from CSR to BSR
//     int base, nnz;
//     int nnzb;
//     int blockDim=2;
//     int m=matrix_dim;
//     int n=matrix_dim;
//     int mb = (m + blockDim-1)/blockDim;
//     int nb = (n + blockDim-1)/blockDim;
//     printf("nb=%d\n",nb);

//     int* bsrRowPtrC;
//     cudaMallocManaged(&bsrRowPtrC, sizeof(int) *(mb+1));
//     cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, m, n,
//         descrA , csrRowPtrA, csrColIndA, blockDim,
//         descrC , bsrRowPtrC, &nnzb);
//     printf("nnzb=========\n");
//     printf("%d\n",nnzb);
//     int* bsrColIndC;
//     double* bsrValC;
//     cudaMallocManaged(&bsrColIndC, sizeof(int)*nnzb);
//     cudaMallocManaged(&bsrValC, sizeof(double)*(blockDim*blockDim)*nnzb);
//     cusparseDcsr2bsr(handle, CUSPARSE_DIRECTION_ROW, m, n,
//         descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
//         descrC, bsrValC, bsrRowPtrC, bsrColIndC);
//     printf("bsrValC=========\n");
//     for(int i=0;i<(blockDim*blockDim)*nnzb;i++)
//     {
//       printf("%f\n",bsrValC[i]);
//     }
//     printf("bsrRowPtrC=========\n");
//     for(int i=0;i<(mb+1);i++)
//     {
//       printf("%d\n",bsrRowPtrC[i]);
//     }
//     printf("bsrColIndC=========\n");
//     for(int i=0;i<nnzb;i++)
//     {
//       printf("%d\n",bsrColIndC[i]);
//     }


//     // step 2: allocate vector x and vector y large enough for bsrmv
//     double *y;
//     cudaMallocManaged(&y, sizeof(double)*(mb*blockDim));
//     // step 3: perform bsrmv
//     double alpha=1;
//     double beta=0;
//     printf("cusparseDbsrmv:%d\n",cusparseStatusCheck(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE , mb, nb, nnzb, &alpha,
//       descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, b, &beta, y)));
//     cudaDeviceSynchronize();
//     printf("y===============\n");
//     for(int i=0;i<mb*blockDim;i++)
//     {
//       printf("%f\n",y[i]);
//     }
    
//     //cuSpare library ends
//     cusparseDestroy(handle);
//     cusparseDestroyMatDescr(descrA);
//     cusparseDestroyMatDescr(descrC);

//     //cuda cudaFree
//     cudaFree(csrValA);
//     cudaFree(csrRowPtrA);
//     cudaFree(csrColIndA);
//     cudaFree(bsrValC);
//     cudaFree(bsrRowPtrC);
//     cudaFree(bsrColIndC);
//     // host memory free
//     delete []A_h;
//     delete []b_h;

//     return 0;
// }
