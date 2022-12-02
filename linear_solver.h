#ifndef linear_solver_h
#define linear_solver_h

#include <vector>

std::vector<float> gaussian_elimination(const std::vector<float> conductance, const int matrix_dim);

#endif