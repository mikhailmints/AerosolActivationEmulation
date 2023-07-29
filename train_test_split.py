import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("in_filename")
parser.add_argument("train_frac")

args = parser.parse_args()

in_filename = args.in_filename
train_frac = float(args.train_frac)

train_filename = in_filename.replace(".csv", "_train.csv")
test_filename = in_filename.replace(".csv", "_test.csv")

df = pd.read_csv(in_filename)

train_size = int(len(df) * train_frac)
perm = np.random.permutation(len(df))
train_df = df.iloc[perm[:train_size]]
test_df = df.iloc[perm[train_size:]]
train_df.to_csv(train_filename, index=False)
test_df.to_csv(test_filename, index=False)

os.remove(in_filename)
