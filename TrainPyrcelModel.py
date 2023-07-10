import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


def get_one_datapoint(df, simulation_id):
    df = df[df["simulation_id"] == simulation_id]
    if df.size == 0:
        return df
    return df[df["S_max"] == np.max(df["S_max"])].sample(1)


dataset_filename = "datasets/dataset1.csv"

df = pd.read_csv(dataset_filename)

# Remove NaN rows
df.dropna(inplace=True)

num_simulations = max(df["simulation_id"]) + 1

# Take 1 datapoint per simulation - one that has highest supersaturation
df = pd.concat([get_one_datapoint(df, i) for i in range(num_simulations)])

# Apply log transformations
df[["mode_N", "mode_mean", "velocity"]] = df[["mode_N", "mode_mean", "velocity"]].apply(
    np.log10
)

# Eliminate outliers
std_threshold = 3
df = df[(np.abs(stats.zscore(df)) < std_threshold).all(axis=1)]

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
        ]
    ]
)
Y = np.array(df["act_frac"])

# nanfilter = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
# X = X[nanfilter]
# Y = Y[nanfilter]

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

# z-score normalization
X = (X - x_mean) / x_std

# Shuffle
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

# unique, inverse, counts = np.unique(
#     np.digitize(Y_train, np.linspace(0, 1, 50)), return_inverse=True, return_counts=True
# )
# weights = (unique / counts)[inverse]

regr = GradientBoostingRegressor(
    learning_rate=0.2, n_estimators=50, max_depth=3, n_iter_no_change=20
)
regr.fit(X_train, Y_train)

r2 = regr.score(X_test, Y_test)
print(f"R^2 = {r2}")


def plot_accuracy_scatterplot(X, Y, ax):
    predictions = regr.predict(X)
    ax.set_xlabel("Aerosol Activation (PySDM)")
    ax.set_ylabel("Aerosol Activation (predicted)")
    ax.plot([0, 1], [0, 1], color="red")
    ax.plot([0, 1], [0, 0.5], color="blue")
    ax.plot([0, 0.5], [0, 1], color="blue")
    ax.scatter(Y, predictions, s=1, color="black")


fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
axs[0].set_title("Model Predictions on Training Data")
axs[1].set_title("Model Predictions on Testing Data")
axs[1].tick_params(labelleft=True)
plot_accuracy_scatterplot(X_train, Y_train, axs[0])
plot_accuracy_scatterplot(X_test, Y_test, axs[1])

plt.show()
