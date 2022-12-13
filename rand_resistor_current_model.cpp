#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"

#include <chrono>




int main(int argc, char *argv[])
{
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
    
    auto begin = std::chrono::high_resolution_clock::now();
    // need to change to the column majored?
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
    fclose(fp);

    double* csrValA=new double[matrix_dim*matrix_dim];
    int* csrRowptrA=new int[matrix_dim]();
    int* csrColIndA=new int[matrix_dim*matrix_dim];
    int nonzeroNums=Dense_to_row_major_CSR(matrix_dim,conductance_definite,csrValA, csrRowptrA,csrColIndA);

    FILE * fp_0;
    fp_0 = fopen ("rand_matrix_CSR.out", "w");
    for(int i=0;i<nonzeroNums;i++)
    {
        fprintf(fp_0,"%f,%d\n",csrValA[i],csrColIndA[i]);
        //printf("%f,%d\n",csrValA[nonzeroNums],csrColIndA[nonzeroNums]);
    }
    for(int i=0;i<matrix_dim;i++)
    {
        fprintf(fp_0,"%d\n",csrRowptrA[i]);
    }
    fclose(fp_0);

    delete []csrValA;
    delete []csrRowptrA;
    delete []csrColIndA;

    gaussian_elimination(conductance_definite, currents_definite, matrix_dim);

    double* voltages=new double[matrix_dim];
    back_substituition(conductance_definite, currents_definite, voltages,matrix_dim);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin);

    printf("%.3f\n", duration_sec.count());


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

    return 0;
}