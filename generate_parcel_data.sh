#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH -J generate_parcel_data
#SBATCH --mail-user=mmints@caltech.edu

## SBATCH --mail-type=BEGIN
## SBATCH --mail-type=END
## SBATCH --mail-type=FAIL


## SBATCH -p general # partition (queue)
#SBATCH -o slurm.out # STDOUT
#SBATCH -e slurm.out # STDERR

echo Starting job

echo NPROCS: $SLURM_NPROCS

mkdir datasets/temp

for ((I=1; I<=$SLURM_NPROCS; I++)) 
do
    srun -n1 --exclusive python3 GenerateParcelData.py datasets/temp/temp_dataset($I).csv --num_simulations=1000 --num_processes=1 &
done

wait

echo Done

