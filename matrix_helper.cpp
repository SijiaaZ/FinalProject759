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
            //printf("%f,%d,%f,%d,%f\n",elementList[i].value,elementList[i].Node1,currents[elementList[i].Node1],elementList[i].Node2,currents[elementList[i].Node2]);
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

void rand_element(Element* elementList, int elementCount,int Node1,int Node2, std::mt19937 & gen, char elementName,std::uniform_real_distribution<double>& current_values,std::uniform_real_distribution<double>& resistor_values)
{
    elementList[elementCount].Node1=Node1;
    elementList[elementCount].Node2=Node2;
    elementList[elementCount].elementName=elementName;
    if(elementList[elementCount].elementName=='I')
    {
        elementList[elementCount].value=current_values(gen);
    }
    else
    {
        elementList[elementCount].value=resistor_values(gen);
    }


    //printf("Name:%c,Node1:%d,Node2:%d,value:%.3f\n",elementList[elementCount].elementName,elementList[elementCount].Node1,elementList[elementCount].Node2,elementList[elementCount].value);
}

void rand_resistor_circuit_model(const int nodeNum, const int elementNum, Element* elementList)
{
    //generate random float values
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_element_choices(0, probability); // min, max of the random float value
    std::uniform_real_distribution<double> current_values(-0.1,0.1);
    std::uniform_real_distribution<double> resistor_values(1,10000);


    int elementCount=0;
    bool srcExist=false;
    //To form a big loop
    for(int i=0;i<nodeNum-1;i++)
    {
        int Node1=i;
        int Node2=i+1;
        char elementName=(dist_element_choices(gen)>(probability-1))?'I':'R';
        rand_element(elementList, elementCount,Node1,Node2, gen, elementName,current_values,resistor_values);
        // if there is two current sources in serial and there is no branch between them, it is not a valid circuit
        // therefore, always add another branch if there is a current source
        elementCount++;
        if(elementName=='I')
        {
            srcExist=true;
            int Node2=rand()%nodeNum;
            rand_element(elementList, elementCount,Node1,Node2, gen, 'R',current_values,resistor_values);
            elementCount++;
        }
        
    }
    

     // connect the head and tail to form the loop
    rand_element(elementList, elementCount,0,nodeNum-1, gen, 'R',current_values,resistor_values);
    elementCount++;
    // To add a few branches
    while(elementCount<elementNum)
    {
        
        int Node1=rand()%nodeNum;
        int Node2=rand()%nodeNum;
        while(Node1==Node2)
        {
            Node2=rand()%nodeNum;
        }
        if(Node1>Node2)
        {
            int temp=Node1;
            Node1=Node2;
            Node2=temp;
        }
        if(!srcExist)
        {
          rand_element(elementList, elementCount,Node1,Node2, gen, 'I',current_values,resistor_values);
          elementCount++;
          continue;
          srcExist=true;
        }
        char elementName=(dist_element_choices(gen)>(probability-1))?'I':'R';
        rand_element(elementList, elementCount,Node1,Node2, gen, elementName,current_values,resistor_values);
        elementCount++;
    }

}


// return the number of non zero elements
int Dense_to_row_major_CSR(int matrix_dim,const double* A,double* csrValA, int* csrRowptrA,int* csrColIndA)
{
    int numzerosNum=0;
    for(int i=0;i<matrix_dim;i++)
    {
        int first_col_num=matrix_dim;
        for(int j=0;j<matrix_dim;j++)
        {
            if(A[i*matrix_dim+j]==0)
                continue;
            if(j<=first_col_num)
            {
              first_col_num=j;
              csrRowptrA[i]=numzerosNum;
            }
            csrValA[numzerosNum]=A[i*matrix_dim+j];
            csrColIndA[numzerosNum]=j;
            //printf("%d,%f,%d,%d\n",numzerosNum,csrValA[numzerosNum],csrColIndA[numzerosNum],csrRowptrA[i]);
            numzerosNum++;
        }
    }
    csrRowptrA[matrix_dim]=numzerosNum;
    return numzerosNum;
}
