#!/bin/bash
#SBATCH --job-name=quicksort
#SBATCH --output=quicksort_output.txt
#SBATCH --error=quicksort_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00


output_file="run_times.txt"
echo "Execution Times" > $output_file
echo "=================" >> $output_file

echo "Running sequential version..." >> $output_file
start_time=$(date +%s%N)
./quicksort_sequential
end_time=$(date +%s%N)
elapsed_time=$((end_time - start_time))
elapsed_time_ms=$((elapsed_time / 1000000))
echo "Sequential version: $elapsed_time_ms ms" >> $output_file

for threads in 2 4 8 16; do
    echo "Running parallel version with $threads threads..." >> $output_file
    export OMP_NUM_THREADS=$threads
    start_time=$(date +%s%N)
    ./quicksort_parallel
    end_time=$(date +%s%N)
    elapsed_time=$((end_time - start_time))
    elapsed_time_ms=$((elapsed_time / 1000000))
    echo "Parallel version with $threads threads: $elapsed_time_ms ms" >> $output_file
done

echo "Execution times have been recorded in $output_file."
