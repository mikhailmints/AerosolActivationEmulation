import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

train_dataset_filename = "datasets/dataset3.csv"

df = pd.read_csv(train_dataset_filename)

X = np.array(
    df[["mode_N", "mode_mean", "mode_kappa", "velocity", "mac", "RH", "T", "p"]]
)
Y = np.array(df["act_frac"])

nanfilter = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X = X[nanfilter]
Y = Y[nanfilter]

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

X = (X - x_mean) / x_std

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

regr = GradientBoostingRegressor(learning_rate=1, max_depth=4)
regr.fit(X_train, Y_train)


def plot_accuracy_scatterplot(X, Y, ax):
    predictions = regr.predict(X)
    ax.set_xlabel("Aerosol Activation (PySDM)")
    ax.set_ylabel("Aerosol Activation (predicted)")
    ax.scatter(Y, predictions, s=1)
    ax.plot([0, max(Y)], [0, max(Y)], color="red", ls="dotted")


fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
axs[0].set_title("Model Predictions on Training Data")
axs[1].set_title("Model Predictions on Testing Data")
axs[1].tick_params(labelleft=True)
plot_accuracy_scatterplot(X_train, Y_train, axs[0])
plot_accuracy_scatterplot(X_test, Y_test, axs[1])

plt.show()
