#ifndef parse_h
#define parse_h
#define MAX_CHAR 256
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>



struct Element
{
    char elementName;
    int Node1;
    int Node2;
    double value;
};

// parseNetlist
// if failed, return NULL, if success return the element array address
// the returned address need to be freed and deleted
Element* parseNetlist(const char* filepath, int & elementListLength);
//parseElement
//if success return 0
int parseElement(char* line, Element& element);

#endif


