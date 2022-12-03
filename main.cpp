#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"
int main(int argc, char *argv[])
{
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
    std::vector<float> conductance(augmented_matrix_dim*augmented_matrix_dim);
    std::vector<float> currents(augmented_matrix_dim);
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
    std::vector<float> conductance_definite((augmented_matrix_dim-1)*(augmented_matrix_dim-1));
    std::vector<float> currents_definite(augmented_matrix_dim-1);
    augmented_Matrix_to_definite_matrix( elementListLength,  conductance,  currents,  conductance_definite, currents_definite,  augmented_matrix_dim);
    printf("definite conductane matrix\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%f,",conductance_definite[i*matrix_dim+j]);
        }
        printf("\n");
    }
    
    
    gaussian_elimination(conductance_definite, currents_definite, matrix_dim);
    printf("conductane matrix after gaussian\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%f ",conductance_definite[i*matrix_dim+j]);
        }
        printf("\n");
    }

    for(int i=0;i<matrix_dim;i++)
    {
        printf("%f\n",currents_definite[i]);
    }

    printf("=======\n");
    printf("Result:\n");
    std::vector<float> voltages=back_substituition(conductance_definite, currents_definite, matrix_dim);
    for(int i=0;i<matrix_dim;i++)
    {
        printf("%f\n",voltages[i]);
    }

    if(elementList)
        delete[] elementList;
    return 0;
}