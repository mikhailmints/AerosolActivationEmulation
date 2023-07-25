from collections import namedtuple
import numpy as np
from AerosolActivationEmulator import AerosolActivationEmulator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor


class AerosolActivationEmulatorXGB(AerosolActivationEmulator):
    def __init__(
        self, dataset_filename, test_frac=0.2, weight_bins=None, weight_sigma=0.2
    ):
        super().__init__(
            dataset_filename,
            test_frac=test_frac,
            weight_bins=weight_bins,
            weight_sigma=weight_sigma,
        )
        self.regr: XGBRegressor = None
        self.hyperparams = {}

    def _predict_act_frac(self, x: np.ndarray):
        assert self.regr is not None
        return self.regr.predict(x)

    def train(self):
        self.regr = XGBRegressor(**self.hyperparams)
        self.regr.fit(self.X_train, self.Y_train, sample_weight=self.train_weights)

    def hyperparameter_search(self, param_distributions, cross_validation_k, n_iter):
        self.regr = XGBRegressor(**self.hyperparams)
        cv = RandomizedSearchCV(
            self.regr,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cross_validation_k,
            verbose=3,
            scoring="neg_mean_squared_error"
        )
        cv.fit(self.X_train, self.Y_train, sample_weight=self.train_weights)
        self.hyperparams = cv.best_params_
