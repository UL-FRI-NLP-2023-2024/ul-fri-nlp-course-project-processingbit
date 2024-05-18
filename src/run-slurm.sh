#!/bin/bash
#SBATCH --job-name=RAG-job                  # Job name
#SBATCH --partition=gpu                     # Partition (queue) name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=24                   # CPU cores/threads per task
#SBATCH --gpus=1                            # Number of GPUs per node
#SBATCH --mem-per-gpu=64G                           # Job memory request
#SBATCH --time=05:00:00                     # Time limit hrs:min:sec
#SBATCH --output=logs/mistral-job_%j.log        # Standard output 
#SBATCH --error=logs/mistral-job_%j.err         # Standard error

singularity exec --nv ./containers/container_llm.sif python3 $1