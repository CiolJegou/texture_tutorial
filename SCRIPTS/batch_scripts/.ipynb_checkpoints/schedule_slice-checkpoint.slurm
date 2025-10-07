#!/bin/bash
# Job name:
#SBATCH --job-name=TexTom
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=slurm_logfiles/slice-reconstruction-%j.out
# Wall clock limit:
#SBATCH --time=00:20:00
#
## Command(s) to run (example):

#module load mamba
#conda activate /home/esrf/xxxxxxx/odftt_env
#source /home/esrf/xxxxxxx/Code/.venv_textom/bin/activate
source /cvmfs/hpc.esrf.fr/software/packages/linux/x86_64/jupyter-slurm/2025.04.3/etc/profile.d/conda.sh
conda activate /home/esrf/ljegou/.conda/envs/mumott_072025

cd $4
python batch_scripts/recon_single_slice.py $1 $2 $3
