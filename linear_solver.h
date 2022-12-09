#ifndef linear_solver_h
#define linear_solver_h

#include <vector>
#include <stdio.h>

void gaussian_elimination(std::vector<double>& conductance_echelon, std::vector<double>& currents_echelon,int matrix_dim);
void gaussian_elimination(double* conductance_echelon, double* currents_echelon,int matrix_dim);
void back_substituition(double* conductance_echelon, double* currents_echelon,double* voltages, int matrix_dim);
std::vector<double> back_substituition(const std::vector<double> conductance_echelon, std::vector<double>& currents_echelon,int matrix_dim);
#endif