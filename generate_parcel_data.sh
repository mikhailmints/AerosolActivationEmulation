#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH -J generate_parcel_data

#SBATCH -o slurm.out # STDOUT
#SBATCH -e slurm.out # STDERR

OUT_FILE=$1
NUM_SIM=$2

echo Starting job

echo NPROCS: $SLURM_NPROCS

rm -rf samples/*

python3 SampleParcelParameters.py --num_simulations=$NUM_SIM --num_processes=$SLURM_NPROCS

mkdir -p datasets/temp

for ((I=1; I<=$SLURM_NPROCS; I++)) 
do
    srun -n1 --exclusive python3 GenerateParcelData.py --out_filename=datasets/temp/temp_dataset($I).csv --sample_filename=samples/sample($I).csv --save_period=5 &
done

wait

python3 CombineTempDataFiles.py $OUT_FILE

rm -rf datasets/temp/

echo Done

