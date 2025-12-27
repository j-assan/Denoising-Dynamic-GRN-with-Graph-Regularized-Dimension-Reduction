#!/bin/bash
#SBATCH --job-name=slurm_grnboost_t20binall
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --output /cluster/scratch/jassan/cardamom/out/slurm_grnboost_t20binall_%j.out
# Print job information

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $PWD"
echo "=========================================="

source ../../venv/bin/activate
python infer_grnboost2_all.py

echo ""
echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="