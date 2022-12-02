// Apapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xpotrf/cusolver_Xpotrf_example.cu

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cusolver_utils.h"

#include "parse.h"


int main(int argc, char *argv[]) {

    int elementListLength=8;
    Element* elementList = (Element*)malloc(elementListLength*sizeof(Element));
    int parseSuccess=parseNetlist("Draft1.txt", elementList,elementListLength);
    printf("Success:%d\n",parseSuccess);
    printf("element List Length:%d\n",elementListLength);
    
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    using data_type = float;

    const int64_t m = 3;
    const int64_t lda = m;
    const int64_t ldb = m;

    /*
     *     | 1     2     3 |
     * A = | 2     5     5 | = L0 * L0**T
     *     | 3     5    12 |
     *
     *            | 1.0000         0         0 |
     * where L0 = | 2.0000    1.0000         0 |
     *            | 3.0000   -1.0000    1.4142 |
     *
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0};
    const std::vector<data_type> B = {1.0, 2.0, 3.0};
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

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    std::printf("=====\n");

    std::printf("B = (matlab base-1)\n");
    print_matrix(m, 1, B.data(), ldb);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
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

    std::printf("after Xpotrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("L = (matlab base-1)\n");
    print_matrix(m, m, L.data(), lda);
    std::printf("=====\n");

    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */

    CUSOLVER_CHECK(cusolverDnXpotrs(cusolverH, NULL, uplo, m, 1, /* nrhs */
                                    traits<data_type>::cuda_data_type, d_A, lda,
                                    traits<data_type>::cuda_data_type, d_B, ldb, d_info));

    CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("X = (matlab base-1)\n");
    print_matrix(m, 1, X.data(), ldb);
    std::printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}