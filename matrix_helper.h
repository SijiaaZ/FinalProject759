#ifndef matrix_helper_h
#define matrix_helper_h
#include <vector>
#include "parse.h"
int get_Matrix_Dim_from_nodes(Element* elementList,int elementListLength);
void elementList_to_Matrix( Element* elementList,int elementListLength, std::vector<float>& conductance, std::vector<float>& currents, int matrix_dim);
#endif

