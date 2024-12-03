#!/bin/bash
#SBATCH --nodelist=fang-s009
#SBATCH --time=00:10:00


./gauss_jacobi input.txt output_2_1024.txt 25000 2 1024
./gauss_jacobi input.txt output_4_512.txt 2500 4 512
./gauss_jacobi input.txt output_8_256.txt 2500 8 256
