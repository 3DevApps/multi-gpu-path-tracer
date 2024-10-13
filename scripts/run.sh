#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plgggpuinz2024proper-gpu
#SBATCH --gpus=8 
#SBATCH -t 01:00:00

cd ~/multi-gpu-path-tracer
source ./scripts/ares_setup.sh 
build/cuda_project "$1" "$2"