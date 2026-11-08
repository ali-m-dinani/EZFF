#!/bin/bash
#SBATCH --job-name=ezff_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --account=acv13_cr_default
#SBATCH --partition=standard

set -euo pipefail
module purge
module load gcc/13.4.0
module load openmpi/5.0.7
module load lammps/20250722
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# reaxkit-worker-contract: 1
exec bash "$1"
