import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("in_dir")
parser.add_argument("out_filename")

args = parser.parse_args()

in_dir = args.in_dir
out_filename = args.out_filename

if os.path.exists(out_filename):
    os.remove(out_filename)

for temp_filename in os.listdir(in_dir):
    temp_filename = os.path.join(in_dir, temp_filename)
    df = pd.read_csv(temp_filename)
    if os.path.exists(out_filename):
        with open(out_filename, "r") as prev_file:
            prev_df = pd.read_csv(prev_file)
        max_prev_id = max(prev_df["simulation_id"])
        df["simulation_id"] += max_prev_id + 1
        df.to_csv(out_filename, mode="a", index=False, header=False)
    else:
        df.to_csv(out_filename, index=False)
