import numpy as np
import sklearn.metrics
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from ExtractParcelFeatures import extract_parcel_features

X, Y, initial_data = extract_parcel_features("datasets/dataset11.csv")
arg_scheme_CliMA = np.array(initial_data["ARG_act_frac_CliMA"])
arg_scheme_pyrcel = np.array(initial_data["ARG_act_frac_pyrcel"])


plt.xscale("log")
plt.xlabel("Aerosol mean radius (m)")
plt.ylabel("(CliMA ARG scheme act frac) - (PySDM act frac)")
plt.scatter(
    initial_data["mode_mean"],
    initial_data["ARG_act_frac_CliMA"] - initial_data["act_frac_S"],
    s=1,
    color="black",
)
plt.show()

perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]
arg_scheme_CliMA = arg_scheme_CliMA[perm]
arg_scheme_pyrcel = arg_scheme_pyrcel[perm]

train_size = int(len(X) * 0.75)

X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

arg_scheme_test = arg_scheme_CliMA[train_size:]


def get_freq_weights(y, bins):
    unique, inverse, counts = np.unique(
        np.digitize(y, np.linspace(0, 1, bins)), return_inverse=True, return_counts=True
    )
    weights = (unique / counts)[inverse]
    return weights


train_weights = get_freq_weights(Y_train, 10)
test_weights = get_freq_weights(Y_test, 10)

regr = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=500, max_depth=5, n_iter_no_change=50
)
regr.fit(X_train, Y_train, sample_weight=train_weights)

train_pred = regr.predict(X_train)
test_pred = regr.predict(X_test)

print(
    f"ARG R^2 = {sklearn.metrics.r2_score(Y_test, arg_scheme_test, sample_weight=test_weights)}"
)
print(
    f"ARG MSE = {sklearn.metrics.mean_squared_error(Y_test, arg_scheme_test, sample_weight=test_weights)}"
)

print(
    f"Model R^2 = {sklearn.metrics.r2_score(Y_test, regr.predict(X_test), sample_weight=test_weights)}"
)
print(
    f"Model MSE = {sklearn.metrics.mean_squared_error(Y_test, test_pred, sample_weight=test_weights)}"
)


def plot_accuracy_scatterplot(ax, Y, predictions):
    ax.set_xlabel("Aerosol Activation (PySDM)")
    ax.set_ylabel("Aerosol Activation (predicted)")
    margin = 0.05
    ax.plot([-margin, 1 + margin], [-margin, 1 + margin], color="red", zorder=1)
    ax.scatter(Y, predictions, s=1, color="black")


fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("ARG Scheme Predictions")
plot_accuracy_scatterplot(ax, Y, arg_scheme_CliMA)

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
axs[0].set_title("Model Predictions on Training Data")
axs[1].set_title("Model Predictions on Testing Data")
axs[1].tick_params(labelleft=True)
plot_accuracy_scatterplot(axs[0], Y_train, regr.predict(X_train))
plot_accuracy_scatterplot(axs[1], Y_test, regr.predict(X_test))
plt.show()
