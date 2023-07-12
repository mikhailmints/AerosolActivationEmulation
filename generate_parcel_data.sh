#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH -n100
#SBATCH -N5
#SBATCH --mem-per-cpu=1G
#SBATCH -J generate_parcel_data

#SBATCH -o slurm.out # STDOUT
#SBATCH -e slurm.err # STDERR

OUT_FILE=$1
NUM_SIM=$2

echo Starting job

echo NPROCS: $SLURM_NPROCS

mkdir -p samples
rm -rf samples/*

python3 SampleParcelParameters.py --num_simulations=$NUM_SIM --num_processes=$SLURM_NPROCS

mkdir -p datasets/temp

for ((I=1; I<=$SLURM_NPROCS; I++)) 
do
    srun -n1 --exclusive python3 GenerateParcelData.py --out_filename="datasets/temp/temp_dataset${I}.csv" --sample_filename="samples/sample${I}.pkl" --save_period=10 &
done

wait

python3 CombineTempDataFiles.py $OUT_FILE

rm -rf datasets/temp/

echo Done

