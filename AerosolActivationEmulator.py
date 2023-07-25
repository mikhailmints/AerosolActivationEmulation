import numpy as np
import pandas as pd
from GenerateParcelData import Z_MAX_PARCEL
from AerosolData import *
from abc import ABCMeta, abstractmethod
from typing import Iterable
import scipy.ndimage


def get_freq_weights(y, num_bins, blur_sigma):
    if len(y) == 0:
        return np.array([])
    bins = np.linspace(0, 1 + np.finfo(np.float64).eps, num_bins)
    unique, inverse, counts = np.unique(
        np.digitize(y, bins), return_inverse=True, return_counts=True
    )
    weights = (unique / counts)[inverse]
    sorting = np.argsort(y)
    weights[sorting] = scipy.ndimage.gaussian_filter1d(
        weights[sorting], blur_sigma * len(weights)
    )
    return weights


class AerosolActivationEmulator(metaclass=ABCMeta):
    def __init__(
        self,
        dataset_filename: str,
        valid_frac=0,
        test_frac=0.2,
        weight_bins=None,
        weight_sigma=0.2,
    ):
        df = pd.read_csv(dataset_filename)

        # Remove datapoints where maximum supersaturation occurred at the very end
        df = df[(df["time"] * df["velocity"] < Z_MAX_PARCEL)]

        self.initial_data = df

        # Apply log transformations
        df[["mode_N", "mode_mean", "velocity"]] = df[
            ["mode_N", "mode_mean", "velocity"]
        ].apply(np.log10)

        self.X = np.array(df[list(AerosolData._fields)])
        self.Y = np.array(df["act_frac_S"])

        # shuffle
        perm = np.random.permutation(len(self.X))
        self.X = self.X[perm]
        self.Y = self.Y[perm]

        # train-validation-test split
        valid_size = int(len(self.X) * valid_frac)
        test_size = int(len(self.X) * test_frac)
        train_size = len(self.X) - valid_size - test_size
        assert train_size > 0 and valid_size >= 0 and test_size >= 0

        self.X_train = self.X[:train_size]
        self.X_valid = self.X[train_size : train_size + valid_size]
        self.X_test = self.X[train_size + valid_size :]
        self.Y_train = self.Y[:train_size]
        self.Y_valid = self.Y[train_size : train_size + valid_size]
        self.Y_test = self.Y[train_size + valid_size :]

        self.x_mean = np.mean(self.X_train, axis=0)
        self.x_std = np.std(self.X_train, axis=0)

        # z-score normalization
        self.X_train = (self.X_train - self.x_mean) / self.x_std
        self.X_valid = (self.X_valid - self.x_mean) / self.x_std
        self.X_test = (self.X_test - self.x_mean) / self.x_std

        if weight_bins is None:
            weight_bins = 1
        self.train_weights = get_freq_weights(self.Y_train, weight_bins, weight_sigma)
        self.valid_weights = get_freq_weights(self.Y_valid, weight_bins, weight_sigma)
        self.test_weights = get_freq_weights(self.Y_test, weight_bins, weight_sigma)

    def aerosol_data_to_vector(self, aerosol_data: AerosolData):
        aerosol_df = pd.DataFrame(aerosol_data._asdict(), index=[0])
        aerosol_df[["mode_N", "mode_mean", "velocity"]] = aerosol_df[
            ["mode_N", "mode_mean", "velocity"]
        ].apply(np.log10)
        result = np.array(aerosol_df)
        result = (result - self.x_mean) / self.x_std
        return result

    @abstractmethod
    def _predict_act_frac(self, x: np.ndarray):
        raise NotImplementedError()

    def predict_act_frac(
        self, aerosol_data: AerosolData | Iterable[AerosolData] | np.ndarray
    ):
        if isinstance(aerosol_data, AerosolData):
            return self.predict_act_frac([aerosol_data])
        elif isinstance(aerosol_data, Iterable) and all(
            isinstance(x, AerosolData) for x in aerosol_data
        ):
            return self._predict_act_frac(
                np.array([self.aerosol_data_to_vector(x) for x in aerosol_data])
            )
        else:
            return self._predict_act_frac(aerosol_data)

    @staticmethod
    def _get_ARG_scheme_data(X):
        return np.array([AerosolData(*x).ARG_act_frac_CliMA for x in X])

    @property
    def ARG_scheme_data(self):
        return self._get_ARG_scheme_data(self.X)

    @property
    def ARG_scheme_data_train(self):
        return self._get_ARG_scheme_data(self.X_train)

    @property
    def ARG_scheme_data_test(self):
        return self._get_ARG_scheme_data(self.X_test)
