#include "matrix_helper.h"

int get_Matrix_Dim_from_nodes(Element* elementList,int elementListLength)
{
    int node_num_max=0;
    for(int i=0;i<elementListLength;i++)
    {
        node_num_max=(node_num_max<elementList[i].Node1)?elementList[i].Node1:node_num_max;
        node_num_max=(node_num_max<elementList[i].Node2)?elementList[i].Node2:node_num_max;
    }
    int matrix_dim=node_num_max+1;
    return matrix_dim;
}
//the input argument conductance matrix should be all zeros
//the input argument currents column matrix should be all zeros
void elementList_to_augmented_Matrix(Element* elementList,int elementListLength, double* conductance, double* currents, int matrix_dim)
{

    for(int i=0;i<elementListLength;i++)
    {
        int row=elementList[i].Node1;
        int col=elementList[i].Node2;
        if(elementList[i].elementName=='R')
        {
            
            conductance[row*matrix_dim+row]+=1/elementList[i].value;
            conductance[col*matrix_dim+col]+=1/elementList[i].value;
            conductance[row*matrix_dim+col]-=1/elementList[i].value;
            conductance[col*matrix_dim+row]-=1/elementList[i].value;
        }
        else if(elementList[i].elementName=='I')
        {
            currents[elementList[i].Node1]-=elementList[i].value;
            currents[elementList[i].Node2]+=elementList[i].value;//flowing in is positive
        }
    }
    // if the diagonal is zero, make it one
    for(int i=0;i<matrix_dim;i++)
    {
        if(conductance[i*matrix_dim+i]==0)
        {
            printf("no connection at that node:%d?\n",i);
            conductance[i*matrix_dim+i]=(double)1;
        }
    }
}

//the input argument conductance matrix should be all zeros
//the input argument currents column matrix should be all zeros
void elementList_to_augmented_Matrix(Element* elementList,int elementListLength, std::vector<double>& conductance, std::vector<double>& currents, int matrix_dim)
{

    for(int i=0;i<elementListLength;i++)
    {
        int row=elementList[i].Node1;
        int col=elementList[i].Node2;
        if(elementList[i].elementName=='R')
        {
            
            conductance[row*matrix_dim+row]+=1/elementList[i].value;
            conductance[col*matrix_dim+col]+=1/elementList[i].value;
            conductance[row*matrix_dim+col]-=1/elementList[i].value;
            conductance[col*matrix_dim+row]-=1/elementList[i].value;
        }
        else if(elementList[i].elementName=='I')
        {
            currents[elementList[i].Node1]-=elementList[i].value;
            currents[elementList[i].Node2]+=elementList[i].value;//flowing in is positive
        }
    }
    // if the diagonal is zero, make it one
    for(int i=0;i<matrix_dim;i++)
    {
        if(conductance[i*matrix_dim+i]==0)
        {
            printf("no connection at that node:%d?\n",i);
            conductance[i*matrix_dim+i]=(double)1;
        }
    }
}


void augmented_Matrix_to_definite_matrix(int elementListLength, const std::vector<double> conductance, const std::vector<double> currents, std::vector<double>& conductance_definite, std::vector<double>& currents_definite, int augmented_matrix_dim)
{
    for(int i=1;i<augmented_matrix_dim;i++)
    {
        for(int j=1;j<augmented_matrix_dim;j++)
        {
            conductance_definite[(i-1)*(augmented_matrix_dim-1)+(j-1)]=conductance[i*augmented_matrix_dim+j];
        }
    }

    for(int i=1;i<augmented_matrix_dim;i++)
    {
        currents_definite[i-1]=currents[i];
    }

}


void augmented_Matrix_to_definite_matrix(int elementListLength, const double* conductance, const double* currents, double* conductance_definite, double* currents_definite, int augmented_matrix_dim)
{
    for(int i=1;i<augmented_matrix_dim;i++)
    {
        for(int j=1;j<augmented_matrix_dim;j++)
        {
            conductance_definite[(i-1)*(augmented_matrix_dim-1)+(j-1)]=conductance[i*augmented_matrix_dim+j];
        }
    }

    for(int i=1;i<augmented_matrix_dim;i++)
    {
        currents_definite[i-1]=currents[i];
    }
}



void rand_resistor_circuit_matrix(const int nodeNum, const int elementNum, double* conductance_definite, double* currents_definite)
{


}