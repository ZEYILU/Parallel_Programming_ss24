#!/bin/bash

#SBATCH --job-name=jacobi_sor_parallel
#SBATCH --output=result_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

g++ -fopenmp -std=c++11 jacobi_parallel.c -o jacobi_parallel
g++ -fopenmp -std=c++11 jacobi.c -o jacobi_seqsequential

for n in 10 100 1000 10000
do
    m=$n 
    for alpha in 0.1 1 10
    do
        for relax in 1 1.5 1.9
        do
            for tol in 1e-6 1e-9
            do
                for mits in 20 200 2000
                do
                    for threads in 2 4 8 16
                    do
                        export OMP_NUM_THREADS=$threads
                        
                        echo "Running: n=$n, m=$m, alpha=$alpha, relax=$relax, tol=$tol, mits=$mits, threads=$threads"           
                        echo "(a) sequential program: "
                        ./jacobi_seqsequential $n $m $alpha $relax $tol $mits
                        echo "(b) parallelized program : "
                        ./jacobi_parallel $n $m $alpha $relax $tol $mits
                        echo ""
                    done
                done
            done
        done
    done
done
