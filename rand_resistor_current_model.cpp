#include "parse.h"
#include "matrix_helper.h"
#include "linear_solver.h"
#include <random>
#include <stdlib.h>

#define probability 7

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

    printf("Name:%c,Node1:%d,Node2:%d,value:%.3f\n",elementList[elementCount].elementName,elementList[elementCount].Node1,elementList[elementCount].Node2,elementList[elementCount].value);
}



int main(int argc, char *argv[])
{
    int nodeNum=std::atoi(argv[1]);
    int elementNum=(int)nodeNum*1.8;
    //generate random float values
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_element_choices(0, probability); // min, max of the random float value
    std::uniform_real_distribution<double> current_values(-0.1,0.1);
    std::uniform_real_distribution<double> resistor_values(1,10000);


    Element* elementList=new Element[elementNum];
    int elementCount=0;
    //To form a big loop
    for(int i=0;i<nodeNum-1;i++)
    {
        int Node1=i;
        int Node2=i+1;
        char elementName=(dist_element_choices(gen)>(probability-1))?'I':'R';
        rand_element(elementList, elementCount,Node1,Node2, gen, elementName,current_values,resistor_values);
        // if there is two current sources in serial and there is no branch between them, it is not a valid circuit
        // therefore, always add another branch if there is a current source
        if(elementName=='I')
        {
            int Node2=rand()%nodeNum;
            rand_element(elementList, elementCount,Node1,Node2, gen, 'R',current_values,resistor_values);
            elementCount++;
        }
        elementCount++;
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
        char elementName=(dist_element_choices(gen)>(probability-1))?'I':'R';
        rand_element(elementList, elementCount,Node1,Node2, gen, elementName,current_values,resistor_values);
        elementCount++;
    }

    int augmented_matrix_dim=get_Matrix_Dim_from_nodes(elementList,elementNum);

    printf("%d\n",augmented_matrix_dim);
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
    for(int i=0;i<augmented_matrix_dim;i++)
    {
        for(int j=0;j<augmented_matrix_dim;j++)
        {
            fprintf(fp,"%f,",conductance[i*augmented_matrix_dim+j]);
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"=================\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            fprintf(fp,"%f,",conductance_definite[i*matrix_dim+j]);
        }
        fprintf(fp,"\n");
    }

    gaussian_elimination(conductance_definite, currents_definite, matrix_dim);
    fprintf(fp,"=================\n");
    for(int i=0;i<matrix_dim;i++)
    {
        for(int j=0;j<matrix_dim;j++)
        {
            fprintf(fp,"%f,",conductance_definite[i*matrix_dim+j]);
        }
        fprintf(fp,"\n");
    }

    double* voltages=new double[matrix_dim];
    back_substituition(conductance_definite, currents_definite, voltages,matrix_dim);

    for(int i=0;i<matrix_dim;i++)
    {
        printf("%f\n",voltages[i]);
    }

    fclose(fp);
    delete []voltages;
    delete []conductance_definite;
    delete []currents_definite;
    delete []elementList;
    delete []conductance;
    delete []currents;

    return 0;
}