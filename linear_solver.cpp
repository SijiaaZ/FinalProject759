#include "linear_solver.h"
void gaussian_elimination(std::vector<float>& conductance_echelon, std::vector<float>& currents_echelon,int matrix_dim)
{
    for(int r=0;r<matrix_dim-1;r++)//the row to be deducted
    {
        for (int i=r;i<matrix_dim-1;i++)
        {
            float coefficient=conductance_echelon[(i+1)*matrix_dim+r]/conductance_echelon[r*matrix_dim+r];
            for(int j=r;j<matrix_dim;j++)
            {
                conductance_echelon[(i+1)*matrix_dim+j]-=coefficient*conductance_echelon[r*matrix_dim+j];
            }
            currents_echelon[i+1]-=coefficient*currents_echelon[r];
        }
    }
}

std::vector<float> back_substituition(const std::vector<float> conductance_echelon, std::vector<float>& currents_echelon,int matrix_dim)
{
    std::vector<float> voltages(matrix_dim);
    for(int r=matrix_dim-1;r>=0;r--)//row to be solved
    {
        for(int c=matrix_dim-1;c>r;c--)
        {
            currents_echelon[r]-=voltages[r+1]*conductance_echelon[r*matrix_dim+c];
        }
        voltages[r]=currents_echelon[r]/conductance_echelon[r*matrix_dim+r];
    }
    return voltages;
}