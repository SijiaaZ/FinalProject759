#!/usr/bin/env zsh
#SBATCH -p wacc
#SBATCH -J gmres
#SBATCH -o gmres.out -e gmres.err
#SBATCH -t 0-00:02:00
#SBATCH --gres=gpu:1


module load nvidia/cuda/11.6.0
nvcc task_gmres.cu gmres.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -o task_gmres
nvprof ./task_gmres