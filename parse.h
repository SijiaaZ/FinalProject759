#define MAX_CHAR 256

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
struct Element
{
    char elementName;
    int Node1;
    int Node2;
    float value;
};

typedef struct Element Element;
// parseNetlist
// elementList should be dynamically allocated using malloc, the malloc size can be any as long as it's positive and nonzero.
// After calling this function, the elementList will be assigned values and elementListLength is the real total number of elements.
// if success return 0
int parseNetlist(const char* filepath, Element* elementList, int & elementListLength);
//parseElement
//if success return 0
int parseElement(char* line, Element& element);