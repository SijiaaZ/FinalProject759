#ifndef matrix_helper_h
#define matrix_helper_h
#include <vector>
#include "parse.h"
int get_Matrix_Dim_from_nodes(Element* elementList,int elementListLength);
void elementList_to_augmented_Matrix( Element* elementList,int elementListLength, std::vector<double>& conductance, std::vector<double>& currents, int matrix_dim);
void augmented_Matrix_to_definite_matrix(int elementListLength, const std::vector<double> conductance, const std::vector<double> currents, std::vector<double>& conductance_definite, std::vector<double>& currents_definite, int augmented_matrix_dim);

#endif

