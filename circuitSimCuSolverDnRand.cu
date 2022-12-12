#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"

int main(int argc, char *argv[]) {

    int nodeNum=std::atoi(argv[1]);
    printf("%d\n",nodeNum);
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
    
    // //cusolver,change A, B and m
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    using data_type = double;

    const int64_t m = matrix_dim;
    const int64_t lda = m;
    const int64_t ldb = m;

    std::vector<data_type> A(matrix_dim*matrix_dim) ;
    std::vector<data_type> B(matrix_dim) ;
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            A[i*matrix_dim+j]=conductance_definite[i*matrix_dim+j];
        }
        B[i]=currents_definite[i];
    }

    std::vector<data_type> X(m, 0);
    std::vector<data_type> L(lda * m, 0);
    int info = 0;

    data_type *d_A = nullptr; /* device copy of A */
    data_type *d_B = nullptr; /* device copy of B */
    int *d_info = nullptr;    /* error info */

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace */

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // std::printf("A = (matlab base-1)\n");
    // print_matrix(m, m, A.data(), lda);
    // std::printf("=====\n");

    // std::printf("B = (matlab base-1)\n");
    // print_matrix(m, 1, B.data(), ldb);
    // std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    auto begin = std::chrono::high_resolution_clock::now();
    cudaEvent_t startEvent, stopEvent; 
    
    cudaEventCreate(&startEvent); 
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);//timing starts

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: query working space */
    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverH, NULL, uplo, m, traits<data_type>::cuda_data_type, d_A, lda,
        traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    /* step 4: Cholesky factorization */
    CUSOLVER_CHECK(cusolverDnXpotrf(cusolverH, NULL, uplo, m, traits<data_type>::cuda_data_type,
                                    d_A, lda, traits<data_type>::cuda_data_type, d_work, d_lwork,
                                    h_work, h_lwork, d_info));

    CUDA_CHECK(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    //std::printf("after Xpotrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    // std::printf("L = (matlab base-1)\n");
    // print_matrix(m, m, L.data(), lda);
    // std::printf("=====\n");

    CUSOLVER_CHECK(cusolverDnXpotrs(cusolverH, NULL, uplo, m, 1, /* nrhs */
                                    traits<data_type>::cuda_data_type, d_A, lda,
                                    traits<data_type>::cuda_data_type, d_B, ldb, d_info));

    CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    cudaEventRecord(stopEvent, 0); 
    cudaEventSynchronize(stopEvent); //wait for that event to complete
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);//in milliseconds

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin);

    cudaEventDestroy(startEvent); 
    cudaEventDestroy(stopEvent);


    printf("%.3f\n", duration_sec.count());

    printf("%.3f\n",elapsedTime);

    
    
    //std::printf("X = (matlab base-1)\n");
    //print_matrix(m, 1, X.data(), ldb);
    FILE * fp_Result_2;
    fp_Result_2 = fopen ("rand_matrix_result_LU_CuSolver.out", "w");
    for (int i = 0; i < m; i++) {
        fprintf(fp_Result_2,"%.3f\n", X.data()[i]);
    }
    fclose(fp_Result_2);

    //std::printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());




    delete []conductance_definite;
    delete []currents_definite;
    delete []elementList;
    delete []conductance;
    delete []currents;

    return EXIT_SUCCESS;
}