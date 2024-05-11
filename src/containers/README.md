# How to build the container
Build the container by running:

```bash
singularity build container_llm.sif container_llm.def
```

and then use run_slurm.sh with:
```bash
sbatch run_slurm.sh
```