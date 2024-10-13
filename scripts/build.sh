#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plgggpuinz2024proper-gpu
#SBATCH --gpus=1
#SBATCH -t 00:15:00
#SBATCH --output=tmp/compile-output.txt
#SBATCH --error=tmp/compile-output.txt

cd ~/multi-gpu-path-tracer

source ./scripts/ares_setup.sh
cmake -S . -B build
cmake --build build "$@"