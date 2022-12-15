#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>

#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"
#include "gmres.h"

int main(int argc, char *argv[]) {
    
    int nodeNum=std::atoi(argv[1]);
    printf("%d\n",nodeNum);
    int elementNum=(int)nodeNum*1.8;
    Element* elementList=new Element[elementNum];

    rand_resistor_circuit_model(nodeNum, elementNum, elementList);

    int augmented_matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementNum);

    int matrix_dim=augmented_matrix_dim-1;
    double* conductance=new double[augmented_matrix_dim*augmented_matrix_dim]();
    double* currents=new double[augmented_matrix_dim]();
    double* conductance_definite=new double[matrix_dim*matrix_dim]();
    double* currents_definite=new double[matrix_dim]();
    
    // need to change to the column majored
    elementList_to_augmented_Matrix(elementList, elementNum, conductance, currents, augmented_matrix_dim);
    augmented_Matrix_to_definite_matrix( elementNum,  conductance,  currents,  conductance_definite, currents_definite,  augmented_matrix_dim);
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
        fprintf(fp,"%f,",currents_definite[i]);
    }
    fprintf(fp,"\n");
    fclose(fp);
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

    auto begin = std::chrono::high_resolution_clock::now();
    GMRES(handle,stream1,A, b, x, Q,  H,matrix_dim,20, 0.001);
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

    cudaFree(A);
    cudaFree(Q);
    cudaFree(H);
    cublasDestroy(handle);
    cudaStreamDestroy(stream1);

    delete []A_h;
    delete []b_h;


    delete []conductance_definite;
    delete []currents_definite;
    delete []conductance;
    delete []currents;

    return EXIT_SUCCESS;
}
