#include "parse.h"


int main(int argc, char *argv[])
{
    int elementListLength=8;
    Element* elementList = (Element*)malloc(elementListLength*sizeof(Element));
    int parseSuccess=parseNetlist("Draft1.txt", elementList,elementListLength);
    printf("Success:%d\n",parseSuccess);
    printf("element List Length:%d\n",elementListLength);
    return 0;
}