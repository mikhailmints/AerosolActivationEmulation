import numpy as np
import pandas as pd
from scipy import stats

def get_one_datapoint(df, simulation_id):
    df = df[df["simulation_id"] == simulation_id]
    if df.size == 0:
        return df
    return df[df["S_max"] == np.max(df["S_max"])].sample(1)

def df_to_X(df):
    pass

def extract_pyrcel_features(dataset_filename):

    df = pd.read_csv(dataset_filename)

    # Remove NaN rows
    df.dropna(inplace=True)

    num_simulations = max(df["simulation_id"]) + 1

    # Take 1 datapoint per simulation - one that has highest supersaturation
    df = pd.concat([get_one_datapoint(df, i) for i in range(num_simulations)])

    # Eliminate outliers
    std_threshold = 3
    df = df[(np.abs(stats.zscore(df)) < std_threshold).all(axis=1)]

    initial_data = df.copy()

    # Apply log transformations
    df[["mode_N", "mode_mean", "velocity"]] = df[["mode_N", "mode_mean", "velocity"]].apply(
        np.log10
    )

    X = np.array(
        df[
            [
                "mode_N",
                "mode_mean",
                "mode_kappa",
                "velocity",
                "mac",
                "qv",
                "RH",
                "T",
                "p",
                "ARG_act_frac"
            ]
        ]
    )
    Y = np.array(df["act_frac"])

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)

    # z-score normalization
    X = (X - x_mean) / x_std

    return X, Y, initial_data

    
