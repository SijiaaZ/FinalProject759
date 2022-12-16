# FinalProject759


## Name
GPU parallel implementation of GMRES for circuit simulation

## Deliverables:

**Code structure**

- Parse.h: 
	- the functions to parse the SPICE netlist files and create an array of struct elementlist

- Matrix_helper.h: 
	- construct circuit matrices from an array of struct elementlist. 
	- randomly generate an array of elementlist.  
	- convert dense matrix format to CSR matrix format. 

- gmres.h: 
	- all functions related to dense format and CSR format gmres implementation

- task_gmres.cu: 
	- dense format GMRES with a random matrix 

- task_gmres_CSR.cu: 
	- CSR format GMRES with a random matrix 

- task_gmres_CSR-spice.cu: 
	- CSR format GMRES with the SPICE netlist file as the input


**Input**
- input to task_gmres_CSR-spice: Draft1.txt Draft2.txt Draft3.txt (SPICE netlist files) 
- input to task_gmres/task_gmres_CSR: the randomly generated circuit elements, the number of which are based on the input arguments


**How to run my code on euler**
- sbatch task_gmres.sh, (./task_gmres [size of random matrix+1])
- sbatch task_gmres_CSR.sh(./task_gmres_CSR [size of random matrix+1])
- sbatch task_gmres_CSR-spice.sh(./task_gmres_CSR-spice [Draft3.txt (spice netlist file)])


**Output files**
- rand_matrix.out: 
	- record the linear equation Ax=b to be solved. 
	- except for the last line: the invertible matrix A (the conductance matrix)
	- the last line: the current vector b
- nodal_voltages.out: 
	- the solution of the linear equations, the nodal voltages vectors, showing the voltage at each node
- gmres.out/gmres_CSR.out: 
	- first line: the size of node nums
	- second line: the least square residual error of the solution
	- third line: the time to solve GMRES


## Authors and acknowledgment
Sijia Zhou
