#!/bin/bash

# --- Slurm Directives ---
#SBATCH --job-name=core_32             # Give your job a name
#SBATCH --output=core_32.log           # Standard output and error log
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --constraint=h200              # Specific node feature (e.g., l40s, h200)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --mem=20GB                     # Memory limit
#SBATCH --time=12:00:00                # Time limit (HH:MM:SS)
#SBATCH --account=torch_pr_717_tandon_advanced
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ajh9498@nyu.edu

singularity exec --nv \
    --overlay /scratch/ajh9498/test/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        echo 'EAGER, 32, 16'
        uv run python -m tests.benchmark_EAGER --bench --bits 32 --num-vars 16 --runs 20 --warmup 3
        echo 'VECT, 32, 16'
        uv run python -m tests.benchmark_VECT --bench --bits 32 --num-vars 16 --runs 20 --warmup 3
        echo 'MONT, 32, 16'
        uv run python -m tests.benchmark_MONT --bench --bits 32 --num-vars 16 --runs 20 --warmup 3
        echo 'COMB, 32, 16'
        uv run python -m tests.benchmark_COMB --bench --bits 32 --num-vars 16 --runs 20 --warmup 3
        echo 'EAGER, 32, 20'
        uv run python -m tests.benchmark_EAGER --bench --bits 32 --num-vars 20 --runs 20 --warmup 3
        echo 'VECT, 32, 20'
        uv run python -m tests.benchmark_VECT --bench --bits 32 --num-vars 20 --runs 20 --warmup 3
        echo 'MONT, 32, 20'
        uv run python -m tests.benchmark_MONT --bench --bits 32 --num-vars 20 --runs 20 --warmup 3
        echo 'COMB, 32, 20'
        uv run python -m tests.benchmark_COMB --bench --bits 32 --num-vars 20 --runs 20 --warmup 3
    "