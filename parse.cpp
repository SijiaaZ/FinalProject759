#include "parse.h"

// Adapted based on: https://github.com/risia/CUDA-SPICE-Circuit-Sim/tree/master/CUDA_SPICE_Circuit_Sim
Element* parseNetlist(const char* filepath, int & elementListLength)
{
    
    elementListLength=0;

    int numlines=0;
    std::ifstream inFileUnused(filepath);
    std::string unused;
    while ( std::getline(inFileUnused, unused) )
        numlines++;
    printf("numlines:%d\n",numlines);
    inFileUnused.close();

    Element* elementList=new Element[numlines];


    std::ifstream inFile(filepath);

    while (inFile)
	{
        char line[MAX_CHAR];
		inFile.getline(line,MAX_CHAR);
        std::cout<<line<<std::endl;
		// Skip comments and empty lines
		if (line[0] == '*' || line[0] == '\0') continue;
		if (line[0] == ';'||line[0]=='.') {
            continue;
		}
		else
        {
            Element element;
            if(parseElement(line,element)!=0)
            {
                return NULL;
            }
            else
            {
                printf("Node1:%d,Node2:%d,value:%.3f\n",element.Node1,element.Node2,element.value);
                elementList[elementListLength]= element;
                elementListLength++;
            }
        }
	}
    return elementList;

}

int parseElement(char* line, Element& element)
{
    
    if(line[0]=='R'||line[0]=='I')
    {
        int tokCount=0;
        element.elementName=line[0];
        char* tok=strtok(line," ");
        while(tok!=NULL)
        {
            tokCount++;
            tok=strtok(NULL," ");
            
            switch(tokCount){
                case 1:
                    if(tok[0]=='N')
                    {
                        tok++;
                    }
                    element.Node1=std::atoi(tok);
                    break;
                case 2:
                    if(tok[0]=='N')
                    {
                        tok++;
                    }
                    element.Node2=std::atoi(tok);
                    break;

                case 3:
                    element.value=std::stod(tok);
                default:
                    break;
            }
        }
         
        //should have four tokens
        if(tokCount!=4)
        {
            return -1;
        }
        return 0;
    }
    return -1;

}
