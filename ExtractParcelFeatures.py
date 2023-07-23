import numpy as np
import pandas as pd
import pyrcel
from GenerateParcelData import Z_MAX_PARCEL


def extract_parcel_features(dataset_filename):
    df = pd.read_csv(dataset_filename)

    df["ARG_act_frac_pyrcel"] = df.apply(
        lambda row: pyrcel.arg2000(
            V=row["velocity"],
            T=row["initial_temperature"],
            P=row["initial_pressure"],
            accom=row["mac"],
            mus=[row["mode_mean"] * 1e6],
            sigmas=[row["mode_stdev"]],
            Ns=[row["mode_N"] / 1e6],
            kappas=[row["mode_kappa"]],
        )[2][0],
        axis=1,
    )

    df = df[(df["time"] * df["velocity"] < Z_MAX_PARCEL)]

    # Save untransformed version of the data
    initial_data = df.copy()

    # Apply log transformations
    df[["mode_N", "mode_mean", "velocity"]] = df[
        ["mode_N", "mode_mean", "velocity"]
    ].apply(np.log10)

    X = np.array(
        df[
            [
                "mode_N",
                "mode_mean",
                "mode_stdev",
                "mode_kappa",
                "velocity",
                "initial_temperature",
                "initial_pressure",
                "ARG_act_frac_CliMA",
            ]
        ]
    )
    Y = np.array(df["act_frac_S"])

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)

    # z-score normalization
    X = (X - x_mean) / x_std

    return X, Y, initial_data
