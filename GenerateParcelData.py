import numpy as np
import os
import time
import timeout_decorator
import argparse
import pickle
import pandas as pd
from PySDM import Formulae
from MyParcelSimulation import MyParcelSimulation
from MyParcelSettings import MyParcelSettings
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal


N_SD = 1000
DZ_PARCEL = 1 * si.s
Z_MAX_PARCEL = 1000 * si.m

SIMULATION_TIMEOUT = 5 * 60

parser = argparse.ArgumentParser()
parser.add_argument("--out_filename", required=True)
parser.add_argument("--sample_filename", required=True)
parser.add_argument("--fail_filename", default=None)
parser.add_argument("--save_period", default=5)
parser.add_argument("--log_filename", default=None)

args = parser.parse_args()

log_filename = args.log_filename


def process_print(s):
    print(
        f"Process {os.getpid()}: {s}",
        file=open(log_filename, "a") if log_filename else None,
    )


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


# Run one parcel simulation, output the data for all timesteps
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
        dt=DZ_PARCEL / velocity,
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
        t_max=Z_MAX_PARCEL / velocity,
        formulae=Formulae(constants={"MAC": mac}),
    )

    initial_params["initial_vapor_mix_ratio"] = settings.initial_vapour_mixing_ratio

    try:
        simulation = MyParcelSimulation(settings)
        results = timeout_decorator.timeout(SIMULATION_TIMEOUT, use_signals=False)(simulation.run)()
    except Exception as err:
        process_print(err)
        initial_params["error"] = str(err).strip("\"'")
        result = pd.DataFrame(initial_params, index=[0])
        return result, False
         
    products = results["products"]
    n_rows = len(list(products.values())[0])

    result = pd.DataFrame(
        {k: [initial_params[k]] * n_rows for k in initial_params.keys()} | products
    )

    return result, True


def generate_data(parameters, out_filename, fail_filename, save_period):
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
        ) = parameters[i]
        try:
            simulation_df, success = generate_data_one_simulation(
                mode_N=10**log_mode_N * si.cm**-3,
                mode_mean=10**log_mode_mean * si.um,
                mode_stdev=mode_stdev,
                mode_kappa=mode_kappa,
                velocity=10**log_velocity * si.m / si.s,
                initial_temperature=initial_temperature * si.kelvin,
                initial_pressure=initial_pressure * si.pascal,
                mac=1,
            )
            # Only take data at the time of maximum supersaturation
            # simulation_df = simulation_df[
            #     simulation_df["S_max"] == np.max(simulation_df["S_max"])
            # ].sample(1)
            simulation_df.insert(0, "simulation_id", i % save_period)
            if success:
                result_df = pd.concat([result_df, simulation_df])
            elif fail_filename:
                save_data(simulation_df, fail_filename)
        except timeout_decorator.TimeoutError:
            process_print("Timed out")
        if i % save_period == save_period - 1:
            save_data(result_df, out_filename)
            result_df = pd.DataFrame()
    save_data(result_df, out_filename)


def main():
    process_print("Starting data generation script")

    out_filename = args.out_filename
    parameters = pickle.load(open(args.sample_filename, "rb"))
    fail_filename = args.fail_filename
    save_period = int(args.save_period)

    generate_data(parameters, out_filename, fail_filename, save_period)

    process_print("Done")


if __name__ == "__main__":
    main()
