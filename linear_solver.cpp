#include "linear_solver.h"
std::vector<float> gaussian_elimination(const std::vector<float> conductance, const int matrix_dim)
{
    std::vector<float> conductance_echelon=conductance;
    for(int r=0;r<matrix_dim-1;r++)
    {
        for (int i=r;i<matrix_dim-1;i++)
        {
            float coefficient=conductance_echelon[(i+1)*matrix_dim+r]/conductance_echelon[r*matrix_dim+r];
            for(int j=r;j<matrix_dim-1;j++)
            {
                conductance_echelon[(i+1)*matrix_dim+j]-=coefficient*conductance_echelon[r*matrix_dim+j];
            }
        }
    }

    return conductance_echelon;


}