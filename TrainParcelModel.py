import numpy as np
import sklearn.metrics
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from ExtractParcelFeatures import extract_parcel_features

X, Y, initial_data = extract_parcel_features("datasets/dataset4.csv")
arg_scheme_CliMA = np.array(initial_data["ARG_act_frac_CliMA"])
arg_scheme_pyrcel = np.array(initial_data["ARG_act_frac_pyrcel"])

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

# unique, inverse, counts = np.unique(
#     np.digitize(Y_train, np.linspace(0, 1, 50)), return_inverse=True, return_counts=True
# )
# weights = (unique / counts)[inverse]

regr = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=500, max_depth=4, n_iter_no_change=50
)
regr.fit(X_train, Y_train)

train_pred = regr.predict(X_train)
test_pred = regr.predict(X_test)

print(f"ARG R^2 = {sklearn.metrics.r2_score(Y_test, arg_scheme_test)}")
print(f"ARG MSE = {sklearn.metrics.mean_squared_error(Y_test, arg_scheme_test)}")

print(f"Model R^2 = {sklearn.metrics.r2_score(Y_test, regr.predict(X_test))}")
print(f"Model MSE = {sklearn.metrics.mean_squared_error(Y_test, test_pred)}")



def plot_accuracy_scatterplot(ax, Y, predictions):
    ax.set_xlabel("Aerosol Activation (PySDM)")
    ax.set_ylabel("Aerosol Activation (predicted)")
    ax.plot([0, 1], [0, 1], color="red")
    ax.plot([0, 1], [0, 0.5], color="blue")
    ax.plot([0, 0.5], [0, 1], color="blue")
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


