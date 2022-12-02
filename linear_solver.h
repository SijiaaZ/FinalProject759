#ifndef linear_solver_h
#define linear_solver_h

#include <vector>

void gaussian_elimination(std::vector<float>& conductance_echelon, std::vector<float>& currents_echelon,int matrix_dim);
std::vector<float> back_substituition(const std::vector<float> conductance_echelon, std::vector<float>& currents_echelon,int matrix_dim);
#endif