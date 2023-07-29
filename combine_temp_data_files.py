import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("in_dir")
parser.add_argument("out_filename")

args = parser.parse_args()

in_dir = args.in_dir
out_filename = args.out_filename

combined_df = pd.DataFrame()

for temp_filename in os.listdir(in_dir):
    temp_filename = os.path.join(in_dir, temp_filename)
    df = pd.read_csv(temp_filename)
    if "simulation_id" in combined_df.keys():
        max_prev_id = max(combined_df["simulation_id"])
        df["simulation_id"] += max_prev_id + 1
    combined_df = pd.concat([combined_df, df])

combined_df.to_csv(out_filename, index=False)
