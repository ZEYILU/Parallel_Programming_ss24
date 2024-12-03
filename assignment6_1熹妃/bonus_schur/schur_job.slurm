#!/bin/bash
#SBATCH --nodelist=fang-s010
#SBATCH --time=00:10:00

mpicxx -fopenmp -o schur schur.c -lm

# Run with 3 processes and 3 threads per process
export OMP_NUM_THREADS=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=3
echo "Running with 3 processes and 3 threads per process"
mpirun -np 3 ./schur schur_input_large.txt output_3_3.txt

# Run with 5 processes and 3 threads per process
export OMP_NUM_THREADS=3
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=3
echo "Running with 5 processes and 3 threads per process"
mpirun -np 5 ./schur schur_input_large.txt output_5_3.txt

# Run with 3 processes and 5 threads per process
export OMP_NUM_THREADS=5
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
echo "Running with 3 processes and 5 threads per process"
mpirun -np 3 ./schur schur_input_large.txt output_3_5.txt

# Run with 7 processes and 2 threads per process
export OMP_NUM_THREADS=2
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
echo "Running with 7 processes and 2 threads per process"
mpirun -np 7 ./schur schur_input_large.txt output_7_2.txt

# Run with 2 processes and 7 threads per process
export OMP_NUM_THREADS=7
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
echo "Running with 2 processes and 7 threads per process"
mpirun -np 2 ./schur schur_input_large.txt output_2_7.txt
