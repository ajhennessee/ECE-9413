#!/bin/bash

#SBATCH --job-name=challenge_64
#SBATCH --output=challenge_64.log
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_717_tandon_advanced
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ajh9498@nyu.edu

singularity exec --nv \
    --overlay /scratch/ajh9498/test/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        echo 'EAGER, 64, 16'
        uv run python -m tests.benchmark_EAGER --bench --bits 64 --num-vars 16 --runs 20 --warmup 3 --enable-challenge64
        echo 'VECT, 64, 16'
        uv run python -m tests.benchmark_VECT --bench --bits 64 --num-vars 16 --runs 20 --warmup 3 --enable-challenge64
        echo 'MONT, 64, 16'
        uv run python -m tests.benchmark_MONT --bench --bits 64 --num-vars 16 --runs 20 --warmup 3 --enable-challenge64
        echo 'COMB, 64, 16'
        uv run python -m tests.benchmark_COMB --bench --bits 64 --num-vars 16 --runs 20 --warmup 3 --enable-challenge64
        echo 'EAGER, 64, 20'
        uv run python -m tests.benchmark_EAGER --bench --bits 64 --num-vars 20 --runs 20 --warmup 3 --enable-challenge64
        echo 'VECT, 64, 20'
        uv run python -m tests.benchmark_VECT --bench --bits 64 --num-vars 20 --runs 20 --warmup 3 --enable-challenge64
        echo 'MONT, 64, 20'
        uv run python -m tests.benchmark_MONT --bench --bits 64 --num-vars 20 --runs 20 --warmup 3 --enable-challenge64
        echo 'COMB, 64, 20'
        uv run python -m tests.benchmark_COMB --bench --bits 64 --num-vars 20 --runs 20 --warmup 3 --enable-challenge64
    "