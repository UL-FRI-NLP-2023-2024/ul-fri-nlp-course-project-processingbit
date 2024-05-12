#!/bin/bash
#SBATCH --job-name=RAG-job                  # Job name
#SBATCH --partition=gpu                     # Partition (queue) name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # CPU cores/threads per task
#SBATCH --gres=gpu:1                        # Number of GPUs per node
#SBATCH --mem=4G                            # Job memory request
#SBATCH --time=02:00:00                     # Time limit hrs:min:sec
#SBATCH --output=logs/RAG-job_%j.log        # Standard output log
#SBATCH --error=logs/RAG-job_%j.err         # Standard error log

singularity exec --nv ./containers/container_llm.sif python3 llm_finetune_lora.py