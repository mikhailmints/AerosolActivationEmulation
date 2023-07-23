#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH -n100
#SBATCH -N5
#SBATCH --mem-per-cpu=5G
#SBATCH -J generate_parcel_data

#SBATCH -o slurm.out # STDOUT
#SBATCH -e slurm.err # STDERR

OUT_FILE=$1 # file to which final results will be written
NUM_SIM=$2 # number of simulations to run in total

FAIL_FILE="${OUT_FILE%.*}_fail.${OUT_FILE##*.}" # file to which all failed runs will be written

echo "Starting job"

echo "NPROCS: $SLURM_NPROCS"
echo "NUM_SIM: $NUM_SIM"
echo "OUT_FILE: $OUT_FILE"
echo "FAIL_FILE: $FAIL_FILE"


# Remove any previous temp files, create new temp directories
rm -rf datasets/temp/
mkdir -p datasets/temp/samples
mkdir -p datasets/temp/success
mkdir -p datasets/temp/fail

# Sample the input parameters, splitting between processes, save to separate pkl files 
python3 SampleParcelParameters.py datasets/temp/samples --num_simulations=$NUM_SIM --num_processes=$SLURM_NPROCS


# Run multiple simulations in parallel
for ((I=1; I<=$SLURM_NPROCS; I++)) 
do
    srun -N1 -n1 --exclusive python3 GenerateParcelData.py \
    --out_filename="datasets/temp/success/temp_dataset${I}.csv" \
    --sample_filename="datasets/temp/samples/sample${I}.pkl" \
    --save_period=5 \
    --log_filename=slurm.out \
    --fail_filename="datasets/temp/fail/temp_dataset${I}_fail.csv" \
    --process_name=$I \
    --simulation_timeout=1000 &
done

wait

# Combine the temp files into final output files
python3 CombineTempDataFiles.py datasets/temp/success $OUT_FILE
python3 CombineTempDataFiles.py datasets/temp/fail $FAIL_FILE


echo "Done"

