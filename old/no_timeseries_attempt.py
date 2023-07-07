import numpy as np
from PySDM import Formulae, products
from MyPyrcelSimulation import MyPyrcelSimulation
from MyPyrcelSettings import MyPyrcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


N_SD = 1000
DT_PARCEL = 0.1 * si.s


def generate_data_one_simulation(
    mode_N,
    mode_mean,
    mode_stdev,
    mode_kappa,
    velocity,
    initial_temperature,
    initial_pressure,
    initial_relative_humidity,
    mac,
):
    print(
        ", ".join(
            map(
                str,
                [
                    mode_N,
                    mode_mean,
                    mode_stdev,
                    mode_kappa,
                    velocity,
                    initial_temperature,
                    initial_pressure,
                    initial_relative_humidity,
                    mac,
                ],
            )
        )
    )
    settings = MyPyrcelSettings(
        dt=DT_PARCEL,
        n_sd_per_mode=(N_SD,),
        aerosol_modes_by_kappa={
            mode_kappa: Lognormal(
                norm_factor=mode_N, m_mode=mode_mean, s_geom=mode_stdev
            )
        },
        vertical_velocity=velocity,
        initial_pressure=initial_pressure,
        initial_temperature=initial_temperature,
        initial_relative_humidity=initial_relative_humidity,
        formulae=Formulae(constants={"MAC": mac}),
    )

    simulation = MyPyrcelSimulation(settings)
    results = simulation.run()
    products = results["products"]

    Y = np.array(products["act_frac"])
    X = np.array(
        [
            [
                mode_N,
                mode_mean,
                mode_stdev,
                mode_kappa,
                velocity,
                mac,
                products["T"][i],
                products["p"][i],
                products["RH"][i],
            ]
            for i in range(len(Y))
        ]
    )

    print(X.shape)
    print(X)
    print(Y.shape)
    print(Y)

    nanfilter = ~np.isnan(X).any(axis=1) | ~np.isnan(Y)
    X = X[nanfilter]
    Y = Y[nanfilter]
    return X, Y


def generate_data(num_simulations):
    param_bounds = [
        (1, 4),  # log10(mode_N)
        (-3, 1),  # log10(mode_mean)
        (1.6, 1.8),  # mode_stdev
        (0, 1.2),  # mode_kappa
        (-2, 1),  # log10(velocity)
        (248, 310),  # temperature
        (50000, 105000),  # pressure
        (1, 1.1), # relative_humidity
        (0.1, 1.0),  # mac
    ]
    sampler = qmc.LatinHypercube(d=len(param_bounds))
    sample = sampler.random(n=num_simulations)
    l_bounds = [b[0] for b in param_bounds]
    u_bounds = [b[1] for b in param_bounds]
    scaled = qmc.scale(sample, l_bounds, u_bounds)
    Xs = []
    Ys = []
    for i in range(num_simulations):
        print("Simulation %d / %d" % (i + 1, num_simulations))
        (
            log_mode_N,
            log_mode_mean,
            mode_stdev,
            mode_kappa,
            log_velocity,
            temperature,
            pressure,
            relative_humidity,
            mac,
        ) = scaled[i]
        try:
            Xi, Yi = generate_data_one_simulation(
                mode_N=10**log_mode_N * si.cm**-3,
                mode_mean=10**log_mode_mean * si.um,
                mode_stdev=mode_stdev,
                mode_kappa=mode_kappa,
                velocity=10**log_velocity * si.m / si.s,
                initial_temperature=temperature * si.kelvin,
                initial_pressure=pressure * si.pascal,
                initial_relative_humidity=relative_humidity,
                mac=mac,
            )
            Xs.append(Xi)
            Ys.append(Yi)
        except RuntimeError as err:
            if str(err) == "Condensation failed":
                print(err)
            else:
                raise
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]
    return X, Y


X_train, Y_train = generate_data(100)
X_test, Y_test = generate_data(10)

regr = GradientBoostingRegressor()
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

