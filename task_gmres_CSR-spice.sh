#!/usr/bin/env zsh
#SBATCH -p wacc
#SBATCH -J gmres
#SBATCH -o gmres_spice-%j.out -e gmres_spice-%j.err
#SBATCH -t 0-00:03:00
#SBATCH --gres=gpu:1


module load nvidia/cuda/11.6.0
module load gcc/9.4.0
nvcc task_gmres_CSR-spice.cu gmres.cu -lcusparse matrix_helper.cpp parse.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -o task_gmres_CSR-spice
#for (( c=8; c<=32768; c*=2 )); do ./task_gmres_CSR $c; done
./task_gmres_CSR-spice Draft3.txt
