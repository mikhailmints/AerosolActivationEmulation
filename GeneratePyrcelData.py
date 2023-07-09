import numpy as np
import os
import time
import argparse
import pandas as pd
import threading
from PySDM import Formulae
from MyPyrcelSimulation import MyPyrcelSimulation
from MyPyrcelSettings import MyPyrcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from scipy.stats import qmc


N_SD = 1000
DT_PARCEL = 0.1 * si.s
T_MAX_PARCEL = 100 * si.s

SAVE_PERIOD = 5

parser = argparse.ArgumentParser()
parser.add_argument("out_filename")
parser.add_argument("--num_simulations", default=1)
parser.add_argument("--num_threads", default=1)

args = parser.parse_args()

out_filename = "datasets/" + args.out_filename
num_simulations = int(args.num_simulations)
num_threads = int(args.num_threads)

file_lock = threading.Lock()


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
    initial_params = locals()

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
    n_rows = len(list(products.values())[0])

    result = pd.DataFrame(
        {k: [initial_params[k]] * n_rows for k in initial_params.keys()} | products
    )

    return result


def save_data(result_df):
    print(f"{threading.current_thread().name}: Saving data to file {out_filename}")
    if (result_df.size == 0):
        print(f"{threading.current_thread().name}: No data to save.")
        return
    while True:
        try:
            file_lock.acquire()
            out_df = result_df.copy()
            if os.path.exists(out_filename):
                with open(out_filename, "r") as prev_file:
                    prev_df = pd.read_csv(prev_file)
                max_prev_id = max(prev_df["simulation_id"])
                out_df["simulation_id"] += max_prev_id + 1
                out_df = pd.concat([prev_df, out_df])
            out_df.to_csv(out_filename, index=False)
        except PermissionError:
            file_lock.release()
            print(
                f"{threading.current_thread().name}: The file is open in another program, cannot write."
            )
            time.sleep(1)
        else:
            file_lock.release()
            print(
                f"{threading.current_thread().name}: Successfully saved data to file."
            )
            break

def sample_parameters(num_simulations):
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
    result = qmc.scale(sample, l_bounds, u_bounds)
    return result

def generate_data(parameters):
    result_df = pd.DataFrame()
    
    for i in range(len(parameters)):
        print(
            f"{threading.current_thread().name}: Simulation {i + 1} / {num_simulations}"
        )
        (
            log_mode_N,
            log_mode_mean,
            mode_stdev,
            mode_kappa,
            log_velocity,
            initial_temperature,
            initial_pressure,
            mac,
        ) = parameters[i]
        try:
            simulation_df = generate_data_one_simulation(
                mode_N=10**log_mode_N * si.cm**-3,
                mode_mean=10**log_mode_mean * si.um,
                mode_stdev=mode_stdev,
                mode_kappa=mode_kappa,
                velocity=10**log_velocity * si.m / si.s,
                initial_temperature=initial_temperature * si.kelvin,
                initial_pressure=initial_pressure * si.pascal,
                mac=mac,
            )
            simulation_df.insert(0, "simulation_id", i % SAVE_PERIOD)
            result_df = pd.concat([result_df, simulation_df])
        except RuntimeError as err:
            if str(err) == "Condensation failed":
                print(err)
            else:
                raise
        if i % SAVE_PERIOD == SAVE_PERIOD - 1:
            save_data(result_df)
            result_df = pd.DataFrame()
    save_data(result_df)


parameters = sample_parameters(num_simulations)

parameters_by_thread = np.array_split(parameters, num_threads)

threads = [
    threading.Thread(
        target=generate_data, args=(parameters_by_thread[i],), name=f"Thread {i + 1}"
    )
    for i in range(num_threads)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
