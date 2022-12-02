#include "parse.h"
#include "matrix_helper.h"
int main(int argc, char *argv[])
{
    int elementListLength=0;
    Element* elementList=NULL;
    int parseSuccess=parseNetlist("Draft1.txt", elementList,elementListLength);
    printf("Success:%d\n",parseSuccess);
    printf("element List Length:%d\n",elementListLength);
    for(int i=0;i<elementListLength;i++)
    {
        printf("Node1:%d,Node2:%d,value:%.3f\n",elementList[i].Node1,elementList[i].Node2,elementList[i].value);
    }

    int matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementListLength);
    printf("%d\n",matrix_dim);
    std::vector<float> conductance(matrix_dim*matrix_dim);
    std::vector<float> currents(matrix_dim*matrix_dim);
    elementList_to_Matrix(elementList, elementListLength, conductance, currents, matrix_dim);
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            printf("%f ",conductance[i*matrix_dim+j]);
        }
        printf("\n");
    }
    for(int i=0;i<matrix_dim;i++)
    {
        printf("%f ",currents[i]);
    }
    printf("\n");
    if(elementListLength>0)
        delete[] elementList;
    return 0;
}