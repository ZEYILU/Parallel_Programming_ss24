#!/bin/bash
#SBATCH --nodelist=fang-s010         
#SBATCH --ntasks=16                   
#SBATCH --cpus-per-task=1             
#SBATCH --time=00:10:00              


mpirun -np 6 ./gauss_jacobi input.txt output_6.txt
mpirun -np 12 ./gauss_jacobi input.txt output_12.txt
mpirun -np 16 ./gauss_jacobi input.txt output_16.txt

