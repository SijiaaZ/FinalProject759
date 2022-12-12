#include <cstdio>
#include <cstdlib>
#include <vector>

//#include <cuda_runtime.h>
//#include <cusolverDn.h>

//#include "cusolver_utils.h"

#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"
#include "gmres.h"

int main(int argc, char *argv[]) {

    int nodeNum=std::atoi(argv[1]);
    int elementNum=(int)nodeNum*1.8;
    Element* elementList=new Element[elementNum];

    rand_resistor_circuit_model(nodeNum, elementNum, elementList);

    int augmented_matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementNum);

    int matrix_dim=augmented_matrix_dim-1;
    double* conductance=new double[augmented_matrix_dim*augmented_matrix_dim];
    double* currents=new double[augmented_matrix_dim];
    double* conductance_definite=new double[matrix_dim*matrix_dim];
    double* currents_definite=new double[matrix_dim];
    
    // need to change to the column majored
    elementList_to_augmented_Matrix(elementList, elementNum, conductance, currents, augmented_matrix_dim);
    augmented_Matrix_to_definite_matrix( elementNum,  conductance,  currents,  conductance_definite, currents_definite,  augmented_matrix_dim);
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
        fprintf(fp,"%f,",currents_definite[i]);
    }
    fprintf(fp,"\n");
    fclose(fp);
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%f,",conductance_definite[i*matrix_dim+j]);
    //     }
    //     printf("\n");
    // }
    
    // //cusolver,change A, B and m
    // cusolverDnHandle_t cusolverH = NULL;
    // cudaStream_t stream = NULL;

    // using data_type = double;

    // const int64_t m = matrix_dim;
    // const int64_t lda = m;
    // const int64_t ldb = m;

    // std::vector<data_type> A(matrix_dim*matrix_dim) ;
    // std::vector<data_type> B(matrix_dim) ;
    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         A[i*matrix_dim+j]=conductance_definite[i*matrix_dim+j];
    //     }
    //     B[i]=currents_definite[i];
    // }

    // for(int i=0;i<matrix_dim;i++)
    // {
    //     for(int j=0;j<matrix_dim;j++)
    //     {
    //         printf("%f,",A[i*matrix_dim+j]);
    //     }
    //     printf("\n");
    // }
    // std::vector<data_type> X(m, 0);
    // std::vector<data_type> L(lda * m, 0);
    // int info = 0;

    // data_type *d_A = nullptr; /* device copy of A */
    // data_type *d_B = nullptr; /* device copy of B */
    // int *d_info = nullptr;    /* error info */

    // size_t d_lwork = 0;     /* size of workspace */
    // void *d_work = nullptr; /* device workspace */
    // size_t h_lwork = 0;     /* size of workspace */
    // void *h_work = nullptr; /* host workspace */

    // cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // // std::printf("A = (matlab base-1)\n");
    // // print_matrix(m, m, A.data(), lda);
    // // std::printf("=====\n");

    // // std::printf("B = (matlab base-1)\n");
    // // print_matrix(m, 1, B.data(), ldb);
    // // std::printf("=====\n");

    // /* step 1: create cusolver handle, bind a stream */
    // CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    // /* step 2: copy A to device */
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
    //                            stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
    //                            stream));

    // /* step 3: query working space */
    // CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
    //     cusolverH, NULL, uplo, m, traits<data_type>::cuda_data_type, d_A, lda,
    //     traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    // /* step 4: Cholesky factorization */
    // CUSOLVER_CHECK(cusolverDnXpotrf(cusolverH, NULL, uplo, m, traits<data_type>::cuda_data_type,
    //                                 d_A, lda, traits<data_type>::cuda_data_type, d_work, d_lwork,
    //                                 h_work, h_lwork, d_info));

    // CUDA_CHECK(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
    //                            stream));
    // CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::printf("after Xpotrf: info = %d\n", info);
    // if (0 > info) {
    //     std::printf("%d-th parameter is wrong \n", -info);
    //     exit(1);
    // }

    // // std::printf("L = (matlab base-1)\n");
    // // print_matrix(m, m, L.data(), lda);
    // // std::printf("=====\n");

    // CUSOLVER_CHECK(cusolverDnXpotrs(cusolverH, NULL, uplo, m, 1, /* nrhs */
    //                                 traits<data_type>::cuda_data_type, d_A, lda,
    //                                 traits<data_type>::cuda_data_type, d_B, ldb, d_info));

    // CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(), cudaMemcpyDeviceToHost,
    //                            stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    
    // //std::printf("X = (matlab base-1)\n");
    // print_matrix(m, 1, X.data(), ldb);
    // FILE * fp_Result_2;
    // fp_Result_2 = fopen ("rand_matrix_result_LU_CuSolver.out", "w");
    // for (int i = 0; i < m; i++) {
    //     fprintf(fp_Result_2,"%.3f\n", X.data()[i]);
    // }
    // fclose(fp_Result_2);

    // //std::printf("=====\n");

    // /* free resources */
    // CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_info));
    // CUDA_CHECK(cudaFree(d_work));

    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    // CUDA_CHECK(cudaStreamDestroy(stream));

    // CUDA_CHECK(cudaDeviceReset());


    // GMRES
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double* A_h=new double[matrix_dim * matrix_dim];
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            A_h[i*matrix_dim+j]=conductance_definite[i*matrix_dim+j];
        }
    }
    double* A;
    cudaMallocManaged(&A, sizeof(double) * matrix_dim * matrix_dim);
    cudaMemcpy(A, A_h, sizeof(double) * matrix_dim * matrix_dim, cudaMemcpyDefault);

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


    GMRES(handle,stream1,A, b, x, Q,  H,matrix_dim,100, 0.001);

    FILE * fp_Result_3;
    fp_Result_3 = fopen ("rand_matrix_result_GMRES_CPU_GPU.out", "w");
    for(int i=0;i<matrix_dim;i++)
    {
        fprintf(fp_Result_3,"%.3f\n",x[i]);
        printf("%.3f\n",x[i]);
    }
    fclose(fp_Result_3);

    cudaFree(A);
    cudaFree(Q);
    cudaFree(H);
    cublasDestroy(handle);
    cudaStreamDestroy(stream1);

    delete []A_h;
    delete []b_h;


    // //gaussian elimination
    gaussian_elimination(conductance_definite, currents_definite, matrix_dim);

    double* voltages=new double[matrix_dim];
    back_substituition(conductance_definite, currents_definite, voltages,matrix_dim);

    FILE * fp_Result_1;
    fp_Result_1 = fopen ("rand_matrix_result_Gaussian_CPU.out", "w");
    for(int i=0;i<matrix_dim;i++)
    {
        fprintf(fp_Result_1,"%.3f\n",voltages[i]);
    }
    fclose(fp_Result_1);

    delete []voltages;
    delete []conductance_definite;
    delete []currents_definite;
    delete []elementList;
    delete []conductance;
    delete []currents;

    return EXIT_SUCCESS;
}