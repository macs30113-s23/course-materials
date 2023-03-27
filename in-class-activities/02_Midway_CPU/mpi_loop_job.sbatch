#!/bin/bash

#SBATCH --job-name=mpi_loop_job
#SBATCH --ntasks=5
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30113

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

for i in {1..5}
do
mpirun -n $i python3 mpi_rand_walk.py >> mpi_loop.out
done