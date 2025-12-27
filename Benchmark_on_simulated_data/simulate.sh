#!/bin/bash
#SBATCH --job-name=slurm_simulate_FN4
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128MB
#SBATCH --output /cluster/scratch/jassan/cardamom/out/slurm_FN4_%j.out
#SBATCH --array=0-9     #Array with 20 Jobs, always 10 running in parallel
# Print job information

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $PWD"
echo "=========================================="
IDX=${SLURM_ARRAY_TASK_ID}
source ../../venv/bin/activate
python simulate_data_FN4.py 10 ${IDX}

echo ""
echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="