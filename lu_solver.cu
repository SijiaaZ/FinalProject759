#include "parse.h"
#include "matrix_helper.h"
#include "lu_factorization.h"
int main(int argc, char *argv[]) {
    char* filename=argv[1];
    int elementListLength=0;
    Element* elementList=parseNetlist(filename, elementListLength);
    printf("Success:%d\n",elementList==NULL);
    printf("element List Length:%d\n",elementListLength);
    for(int i=0;i<elementListLength;i++)
    {
        printf("Node1:%d,Node2:%d,value:%.3f\n",elementList[i].Node1,elementList[i].Node2,elementList[i].value);
    }
    int augmented_matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementListLength);
    printf("%d\n",augmented_matrix_dim);
    double* conductance=new double[augmented_matrix_dim*augmented_matrix_dim];
    double* currents=new double[augmented_matrix_dim*augmented_matrix_dim];

    elementList_to_augmented_Matrix(elementList, elementListLength, conductance, currents, augmented_matrix_dim);
    for(int i=0;i<augmented_matrix_dim;i++)
    {
        for(int j=0;j<augmented_matrix_dim;j++)
        {
            printf("%f,",conductance[i*augmented_matrix_dim+j]);
        }
        printf("\n");
    }
    for(int i=0;i<augmented_matrix_dim;i++)
    {
        printf("%f\n",currents[i]);
    }
    int matrix_dim=augmented_matrix_dim-1;
    double* conductance_definite=new double[matrix_dim*matrix_dim];
    double* currents_definite=new double[matrix_dim*matrix_dim];
    augmented_Matrix_to_definite_matrix( elementListLength,  conductance,  currents,  conductance_definite, currents_definite,  augmented_matrix_dim);


    double* L_h=new double[matrix_dim*matrix_dim]();
    double* U_h=new double[matrix_dim*matrix_dim]();
    for(int i=0;i<matrix_dim;i++)
    {
        L_h[i*matrix_dim+i]=(double)1;
    }

    double* L_d;
    double* U_d;
    double* A_d;
    
    cudaMalloc((void**)&L_d, sizeof(double) * matrix_dim*matrix_dim);
    cudaMalloc((void**)&U_d, sizeof(double) * matrix_dim*matrix_dim);
    cudaMalloc((void**)&A_d, sizeof(double) * matrix_dim*matrix_dim);

    cudaMemcpy(A_d, conductance_definite, sizeof(double) * matrix_dim*matrix_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(L_d, L_h, sizeof(double) * matrix_dim*matrix_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(U_d, U_h, sizeof(double) * matrix_dim*matrix_dim, cudaMemcpyHostToDevice);


    LU_factorization_GPU(A_d,L_d,U_d,matrix_dim);

    cudaMemcpy(L_h, L_d, sizeof(double) *  matrix_dim*matrix_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(U_h, U_d, sizeof(double) *  matrix_dim*matrix_dim, cudaMemcpyDeviceToHost);

    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%f,",L_h[i*matrix_dim+j]);
        }
        printf("\n");
    }
    printf("========\n");

    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%f,",U_h[i*matrix_dim+j]);
        }
        printf("\n");
    }
    
    

    if(elementList)
        delete[] elementList;

    delete[] L_h;
    delete[] U_h;
    delete[] conductance;
    delete[] currents;
    delete[] currents_definite;
    delete[] conductance_definite;

    return 0;


}

