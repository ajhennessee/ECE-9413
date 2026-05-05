#!/bin/bash

#SBATCH --job-name=cpu_32
#SBATCH --output=cpu_32.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_717_tandon_advanced
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ajh9498@nyu.edu

singularity exec \
    --overlay /scratch/ajh9498/test/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        export JAX_PLATFORMS=cpu
        echo 'EAGER, CPU, 32, 16'
        uv run python -m tests.benchmark_EAGER --bench --bits 32 --num-vars 16 --runs 20 --warmup 3 --enable-challenge32 2>&1
        echo 'EAGER, CPU, 32, 20'
        uv run python -m tests.benchmark_EAGER --bench --bits 32 --num-vars 20 --runs 20 --warmup 3 --enable-challenge32 2>&1
    "