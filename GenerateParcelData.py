import numpy as np
import os
import time
import argparse
import pickle
import pandas as pd
from PySDM import Formulae
from MyParcelSimulation import MyParcelSimulation
from MyParcelSettings import MyParcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal


N_SD = 1000
DT_PARCEL = 0.1 * si.s
T_MAX_PARCEL = 100 * si.s


def process_print(s):
    print(f"Process {os.getpid()}: {s}")


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


def save_data(result_df, out_filename):
    process_print(f"Saving data to file {out_filename}")
    if result_df.size == 0:
        process_print("No data to save.")
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


def generate_data(parameters, out_filename, save_period):
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
            simulation_df.insert(0, "simulation_id", i % save_period)
            result_df = pd.concat([result_df, simulation_df])
        except RuntimeError as err:
            if str(err) == "Condensation failed":
                process_print(err)
            else:
                raise
        if i % save_period == save_period - 1:
            save_data(result_df, out_filename)
            result_df = pd.DataFrame()
    save_data(result_df, out_filename)


def main():
    process_print("Starting data generation script")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_filename", required=True)
    parser.add_argument("--sample_filename", required=True)
    parser.add_argument("--save_period", default=5)

    args = parser.parse_args()

    out_filename = args.out_filename
    parameters = pickle.load(open(args.sample_filename, "rb"))
    save_period = int(args.save_period)

    generate_data(parameters, out_filename, save_period)

    process_print("Done")


if __name__ == "__main__":
    main()
