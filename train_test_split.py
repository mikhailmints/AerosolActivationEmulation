import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("in_filename")
parser.add_argument("train_filename")
parser.add_argument("test_filename")
parser.add_argument("train_frac")

args = parser.parse_args()

in_filename = args.in_filename
train_filename = args.train_filename
test_filename = args.test_filename
train_frac = float(args.train_frac)

df = pd.read_csv(in_filename)

train_size = int(len(df) * train_frac)
perm = np.random.permutation(len(df))
train_df = df.iloc[perm[:train_size]]
test_df = df.iloc[perm[train_size:]]
train_df.to_csv(train_filename, index=False)
test_df.to_csv(test_filename, index=False)

os.remove(in_filename)
