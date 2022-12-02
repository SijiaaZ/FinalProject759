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
void elementList_to_Matrix(Element* elementList,int elementListLength, std::vector<float>& conductance, std::vector<float>& currents, int matrix_dim)
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
            currents[elementList[i].Node1]+=elementList[i].value;
            currents[elementList[i].Node2]-=elementList[i].value;
            conductance[elementList[i].Node1*matrix_dim+elementList[i].Node1]+=(float)1;
            conductance[elementList[i].Node2*matrix_dim+elementList[i].Node2]+=(float)1;
            conductance[row*matrix_dim+col]=(float)(-1);
            conductance[col*matrix_dim+row]=(float)(-1);
        }
    }
    // if the diagonal is zero, make it one
    for(int i=0;i<matrix_dim;i++)
    {
        if(conductance[i*matrix_dim+i]==0)
        {
            conductance[i*matrix_dim+i]=(float)1;
        }
    }
}

