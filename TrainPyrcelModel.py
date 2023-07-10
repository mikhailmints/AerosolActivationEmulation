import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from ExtractPyrcelFeatures import extract_pyrcel_features

X, Y, initial_data = extract_pyrcel_features("datasets/dataset1.csv")
arg_scheme = np.array(initial_data["ARG_act_frac"])

# Shuffle
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]
arg_scheme = arg_scheme[perm]

train_size = int(len(X) * 0.5)

X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

arg_scheme_test = arg_scheme[train_size:]

# unique, inverse, counts = np.unique(
#     np.digitize(Y_train, np.linspace(0, 1, 50)), return_inverse=True, return_counts=True
# )
# weights = (unique / counts)[inverse]

regr = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=100, max_depth=3, n_iter_no_change=20
)
regr.fit(X_train, Y_train)

r2 = regr.score(X_test, Y_test)
print(f"R^2 = {r2}")


def plot_accuracy_scatterplot(ax, Y, predictions):
    ax.set_xlabel("Aerosol Activation (PySDM)")
    ax.set_ylabel("Aerosol Activation (predicted)")
    ax.plot([0, 1], [0, 1], color="red")
    ax.plot([0, 1], [0, 0.5], color="blue")
    ax.plot([0, 0.5], [0, 1], color="blue")
    ax.scatter(Y, predictions, s=1, color="black")

fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("ARG Scheme Predictions on Testing Data")
plot_accuracy_scatterplot(ax, Y_test, arg_scheme_test)

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
axs[0].set_title("Model Predictions on Training Data")
axs[1].set_title("Model Predictions on Testing Data")
axs[1].tick_params(labelleft=True)
plot_accuracy_scatterplot(axs[0], Y_train, regr.predict(X_train))
plot_accuracy_scatterplot(axs[1], Y_test, regr.predict(X_test))
plt.show()


