#!/bin/bash
# Job name:
#SBATCH --job-name=rad_integ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logfiles/z2_redo-%j.out
#
# Wall clock limit:
#SBATCH --time=01:30:00
#
## Command(s) to run (example):

cd /gpfs/easy/data/id11/inhouse1/Texture_workflow/texture_tutorial/SCRIPTS
python batch_scripts/z2_redo.py $1
