#include "linear_solver.h"
void gaussian_elimination(std::vector<double>& conductance_echelon, std::vector<double>& currents_echelon,int matrix_dim)
{
    for(int r=0;r<matrix_dim-1;r++)//the row to be substracted
    {
        for (int i=r;i<matrix_dim-1;i++)// the row to be substracted from
        {
            double coefficient=conductance_echelon[(i+1)*matrix_dim+r]/conductance_echelon[r*matrix_dim+r];
            for(int j=r;j<matrix_dim;j++)
            {
                conductance_echelon[(i+1)*matrix_dim+j]-=coefficient*conductance_echelon[r*matrix_dim+j];
            }
            currents_echelon[i+1]-=coefficient*currents_echelon[r];
        }
    }
}

std::vector<double> back_substituition(const std::vector<double> conductance_echelon, std::vector<double>& currents_echelon,int matrix_dim)
{
    std::vector<double> voltages(matrix_dim);
    for(int r=matrix_dim-1;r>=0;r--)//row to be solved
    {
        for(int c=matrix_dim-1;c>r;c--)
        {
            currents_echelon[r]-=voltages[c]*conductance_echelon[r*matrix_dim+c];
        }
        voltages[r]=currents_echelon[r]/conductance_echelon[r*matrix_dim+r];
    }
    return voltages;
}
void gaussian_elimination(double* conductance_echelon, double* currents_echelon,int matrix_dim)
{
    for(int r=0;r<matrix_dim-1;r++)//the row to be substracted
    {
        for (int i=r;i<matrix_dim-1;i++)// the row to be substracted from
        {
            double coefficient=conductance_echelon[(i+1)*matrix_dim+r]/conductance_echelon[r*matrix_dim+r];
            for(int j=r;j<matrix_dim;j++)
            {
                conductance_echelon[(i+1)*matrix_dim+j]-=coefficient*conductance_echelon[r*matrix_dim+j];
            }
            currents_echelon[i+1]-=coefficient*currents_echelon[r];
        }
    }
}
void back_substituition(double* conductance_echelon, double* currents_echelon,double* voltages, int matrix_dim)
{
    for(int r=matrix_dim-1;r>=0;r--)//row to be solved
    {
        for(int c=matrix_dim-1;c>r;c--)
        {
            currents_echelon[r]-=voltages[c]*conductance_echelon[r*matrix_dim+c];
        }
        voltages[r]=currents_echelon[r]/conductance_echelon[r*matrix_dim+r];
    }
}