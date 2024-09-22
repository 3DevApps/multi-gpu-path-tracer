#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plgggpuinz2024proper-gpu
#SBATCH --gpus=2 
#SBATCH -t 00:20:00
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd ~/multi-gpu-path-tracer
rm output.out error.err

source ./scripts/ares_setup.sh 
cmake -S . -B build
cmake --build build

build/cuda_project "$1" "$2"
