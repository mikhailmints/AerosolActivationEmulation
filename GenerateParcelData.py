import numpy as np
import os
import time
import argparse
import pandas as pd
import multiprocessing as mp
from PySDM import Formulae
from MyParcelSimulation import MyParcelSimulation
from MyParcelSettings import MyParcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from scipy.stats import qmc


N_SD = 1000
DT_PARCEL = 0.1 * si.s
T_MAX_PARCEL = 100 * si.s

SAVE_PERIOD = 5


def process_print(*args, **kwargs):
    print(f"Process {mp.current_process().name} (PID {os.getpid()}): ", end="")
    print(*args, **kwargs)


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

    settings = MyParcelSettings(
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

    simulation = MyParcelSimulation(settings)
    results = simulation.run()
    products = results["products"]
    n_rows = len(list(products.values())[0])

    result = pd.DataFrame(
        {k: [initial_params[k]] * n_rows for k in initial_params.keys()} | products
    )
    result = result[result["S_max"] == np.max(result["S_max"])].sample(1)

    return result


def save_data(result_df, out_filename, file_lock):
    file_lock.acquire()
    process_print(f"Saving data to file {out_filename}")
    if result_df.size == 0:
        process_print("No data to save.")
        file_lock.release()
        return
    while True:
        try:
            out_df = result_df.copy()
            if os.path.exists(out_filename):
                with open(out_filename, "r") as prev_file:
                    prev_df = pd.read_csv(prev_file)
                max_prev_id = max(prev_df["simulation_id"])
                out_df["simulation_id"] += max_prev_id + 1
                out_df.to_csv(out_filename, mode="a", index=False, header=False)
            else:
                out_df.to_csv(out_filename, index=False)
        except PermissionError:
            process_print("The file is open in another program, cannot write.")
            time.sleep(1)
        else:
            process_print("Successfully saved data to file.")
            break
    file_lock.release()


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


def generate_data(parameters, out_filename, file_lock):
    result_df = pd.DataFrame()

    num_simulations = len(parameters)

    for i in range(len(parameters)):
        process_print(f"Simulation {i + 1} / {num_simulations}")
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
                process_print(err)
            else:
                raise
        if i % SAVE_PERIOD == SAVE_PERIOD - 1:
            save_data(result_df, out_filename, file_lock)
            result_df = pd.DataFrame()
    save_data(result_df, out_filename, file_lock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_filename")
    parser.add_argument("--num_simulations", default=1)
    parser.add_argument("--num_processes", default=1)

    args = parser.parse_args()

    out_filename = "datasets/" + args.out_filename
    num_simulations = int(args.num_simulations)
    num_processes = int(args.num_processes)

    parameters = sample_parameters(num_simulations)

    parameters_by_process = np.array_split(parameters, num_processes)

    file_lock = mp.Lock()

    processes = [
        mp.Process(
            target=generate_data,
            args=(parameters_by_process[i], out_filename, file_lock),
            name=f"{i + 1}",
        )
        for i in range(num_processes)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
