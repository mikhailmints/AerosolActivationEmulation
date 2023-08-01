#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH -n300
#SBATCH -N10
#SBATCH --mem-per-cpu=5G
#SBATCH -J generate_parcel_data

#SBATCH -o slurm.out
#SBATCH -e /dev/null

OUT_NAME=$1
NUM_SIM=$2   # number of simulations to run in total
NUM_MODES=$3 # number of aerosol distribution modes

TRAIN_FILE="datasets/${OUT_NAME}_train.csv"
TEST_FILE="datasets/${OUT_NAME}_test.csv"
FAIL_FILE="datasets/${OUT_NAME}_fail.csv"

echo "Starting job"

echo "NPROCS: $SLURM_NPROCS"
echo "NUM_SIM: $NUM_SIM"

# Remove any previous temp files, create new temp directories
rm -rf datasets/temp/
mkdir -p datasets/temp/samples
mkdir -p datasets/temp/success
mkdir -p datasets/temp/fail

echo "Sampling parameters"

# Sample the input parameters, splitting between processes, save to separate pkl files
python3 sample_parcel_parameters.py datasets/temp/samples --num_simulations=$NUM_SIM --num_modes=$NUM_MODES --num_processes=$SLURM_NPROCS

echo "Starting simulations"

# Run multiple simulations in parallel
for ((I = 1; I <= $SLURM_NPROCS; I++)); do
    srun -N1 -n1 --exclusive python3 generate_parcel_data.py \
        --out_filename="datasets/temp/success/temp_dataset${I}.csv" \
        --sample_filename="datasets/temp/samples/sample${I}.pkl" \
        --save_period=5 \
        --log_filename=slurm.out \
        --fail_filename="datasets/temp/fail/temp_dataset${I}_fail.csv" \
        --process_name=$I \
        --simulation_timeout=300 &
done

while true; do
    NUM_DONE=$(cat slurm.out | grep "Process [0-9]* (PID [0-9]*): Done" | wc -l)
    if [ $NUM_DONE == $SLURM_NPROCS ]; then
        break
    fi
    echo "Still waiting for processes to finish. $NUM_DONE / $SLURM_NPROCS are done."
    sleep 60
done

echo "Combining data files"

# Combine the temp files into final output files
python3 combine_temp_data_files.py datasets/temp/success "datasets/${OUT_NAME}.csv"
python3 train_test_split.py "datasets/${OUT_NAME}.csv" $TRAIN_FILE $TEST_FILE 0.8
python3 combine_temp_data_files.py datasets/temp/fail $FAIL_FILE

echo "Done"
