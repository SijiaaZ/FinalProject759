#ifndef matrix_helper_h
#define matrix_helper_h
#include <vector>
#include "parse.h"
#define probability 7
int get_Matrix_Dim_from_nodes(Element* elementList,int elementListLength);
void elementList_to_augmented_Matrix( Element* elementList,int elementListLength, std::vector<double>& conductance, std::vector<double>& currents, int matrix_dim);
void elementList_to_augmented_Matrix( Element* elementList,int elementListLength,  double* conductance,  double* currents, int matrix_dim);
void augmented_Matrix_to_definite_matrix(int elementListLength, const std::vector<double> conductance, const std::vector<double> currents, std::vector<double>& conductance_definite, std::vector<double>& currents_definite, int augmented_matrix_dim);
void augmented_Matrix_to_definite_matrix(int elementListLength, const double* conductance, const double* currents, double* conductance_definite, double* currents_definite, int augmented_matrix_dim);
#endif

