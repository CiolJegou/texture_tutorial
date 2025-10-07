#!/bin/bash
# Job name:
#SBATCH --job-name=rad_integ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logfiles/slurm-%j.out
#
# Wall clock limit:
#SBATCH --time=01:30:00
#
## Command(s) to run (example):

cd /gpfs/ga/data/visitor/ihma652/id11/20250408/Texture_workflow
python integrate_one_projection.py $1
