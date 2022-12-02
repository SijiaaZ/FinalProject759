#ifndef matrix_helper_h
#define matrix_helper_h
#include <vector>
#include "parse.h"
int get_Matrix_Dim_from_nodes(Element* elementList,int elementListLength);
void elementList_to_augmented_Matrix( Element* elementList,int elementListLength, std::vector<float>& conductance, std::vector<float>& currents, int matrix_dim);
void augmented_Matrix_to_definite_matrix(int elementListLength, const std::vector<float> conductance, const std::vector<float> currents, std::vector<float>& conductance_definite, std::vector<float>& currents_definite, int augmented_matrix_dim);

#endif

