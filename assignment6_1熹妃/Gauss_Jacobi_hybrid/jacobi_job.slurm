#!/bin/bash
#SBATCH --nodelist=fang-s012
#SBATCH --time=00:10:00

mpicc -fopenmp -o jacobi_hybrid jacobi_hybrid.c -lm

export OMP_NUM_THREADS=3

#SBATCH --ntasks=3
#SBATCH --cpus-per-task=3
echo "Running with 3 processes and 3 threads per process"
mpirun -np 3 ./jacobi_hybrid input.txt output_3_3.txt 1000 0.000001

#SBATCH --ntasks=5
#SBATCH --cpus-per-task=3
echo "Running with 5 processes and 3 threads per process"
mpirun -np 5 ./jacobi_hybrid input.txt output_5_3.txt 1000 0.000001

#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
export OMP_NUM_THREADS=5
echo "Running with 3 processes and 5 threads per process"
mpirun -np 3 ./jacobi_hybrid input.txt output_3_5.txt 1000 0.000001

#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
export OMP_NUM_THREADS=2
echo "Running with 7 processes and 2 threads per process"
mpirun -np 7 ./jacobi_hybrid input.txt output_7_2.txt 1000 0.000001

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
export OMP_NUM_THREADS=7
echo "Running with 2 processes and 7 threads per process"
mpirun -np 2 ./jacobi_hybrid input.txt output_2_7.txt 1000 0.000001
