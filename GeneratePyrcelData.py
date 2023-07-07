import numpy as np
import pickle
import os
import sys
from PySDM import Formulae
from MyPyrcelSimulation import MyPyrcelSimulation
from MyPyrcelSettings import MyPyrcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from scipy.stats import qmc
import matplotlib.pyplot as plt


N_SD = 1000
DT_PARCEL = 0.1 * si.s
T_MAX_PARCEL = 100 * si.s

SAVE_PERIOD = 10

out_filename = "datasets/" + sys.argv[1]
num_simulations = int(sys.argv[2])

prev_contents = None

if os.path.exists(out_filename):
    while True:
        choice = input(
            "File already exists. (O)verwrite previous contents, \
(A)dd to previous contents, or (C)ancel? "
        )
        if choice == "O":
            break
        elif choice == "A":
            with open(out_filename, "rb") as prev_file:
                prev_contents = pickle.load(prev_file)
            break
        elif choice == "C":
            exit(0)


def generate_data_one_simulation(
    mode_N,
    mode_mean,
    mode_stdev,
    mode_kappa,
    velocity,
    initial_temperature,
    initial_pressure,
    mac,
):

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
        initial_relative_humidity=1,
        t_max=T_MAX_PARCEL,
        formulae=Formulae(constants={"MAC": mac}),
    )

    simulation = MyPyrcelSimulation(settings)
    results = simulation.run()
    products = results["products"]

    Y = np.array(products["act_frac"])
    X = np.array(
        [
            [
                np.log(mode_N),
                np.log(mode_mean),
                mode_stdev,
                mode_kappa,
                np.log(velocity),
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

    nanfilter = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
    X = X[nanfilter]
    Y = Y[nanfilter]

    return X, Y


def save_data(Xs, Ys):
    print(f"Saving data to file {out_filename}")
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    if prev_contents:
        prev_X, prev_Y = prev_contents
        X = np.append(prev_X, X, axis=0)
        Y = np.append(prev_Y, Y, axis=0)
    with open(out_filename, "wb") as outfile:
        pickle.dump((X, Y), outfile)


def generate_data(num_simulations):
    param_bounds = [
        (1, 4),  # log10(mode_N)
        (-3, 1),  # log10(mode_mean)
        (1.6, 1.8),  # mode_stdev
        (0, 1.2),  # mode_kappa
        (-2, 1),  # log10(velocity)
        (248, 310),  # initial_temperature
        (50000, 105000),  # initial_pressure
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
            initial_temperature,
            initial_pressure,
            mac,
        ) = scaled[i]
        try:
            Xi, Yi = generate_data_one_simulation(
                mode_N=10**log_mode_N * si.cm**-3,
                mode_mean=10**log_mode_mean * si.um,
                mode_stdev=mode_stdev,
                mode_kappa=mode_kappa,
                velocity=10**log_velocity * si.m / si.s,
                initial_temperature=initial_temperature * si.kelvin,
                initial_pressure=initial_pressure * si.pascal,
                mac=mac,
            )
            Xs.append(Xi)
            Ys.append(Yi)
        except RuntimeError as err:
            if str(err) == "Condensation failed":
                print(err)
            else:
                raise
        if i % SAVE_PERIOD == SAVE_PERIOD - 1:
            save_data(Xs, Ys)
    save_data(Xs, Ys)


generate_data(num_simulations)
