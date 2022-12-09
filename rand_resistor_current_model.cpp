#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"






int main(int argc, char *argv[])
{
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
    fp = fopen ("rand_matrix.txt", "w");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            fprintf(fp,"%f,",conductance_definite[i*matrix_dim+j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
    gaussian_elimination(conductance_definite, currents_definite, matrix_dim);

    double* voltages=new double[matrix_dim];
    back_substituition(conductance_definite, currents_definite, voltages,matrix_dim);

    FILE * fp_Result_1;
    fp_Result_1 = fopen ("rand_matrix_result_Gaussian_CPU.txt", "w");
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