#!/bin/bash
#SBATCH --nodelist=fang-s010         
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5    
#SBATCH --time=00:10:00 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 记录开始时间
start_time=$(date +%s)

# 运行程序
mpiexec -np $SLURM_NTASKS ./hybrid_program input.txt output_3_5.txt

# 记录结束时间
end_time=$(date +%s)

# 计算并记录运行时间
runtime=$((end_time - start_time))
echo "Runtime for $SLURM_NTASKS processes and $OMP_NUM_THREADS threads: $runtime seconds" >> output_3_5.txt