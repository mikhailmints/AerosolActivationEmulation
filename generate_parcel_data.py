import numpy as np
import os
import time
import timeout_decorator
import argparse
import pickle
import pandas as pd
from my_parcel_simulation import MyParcelSimulation
from my_parcel_settings import MyParcelSettings
from PySDM.physics import si


N_SD = 100
DZ_PARCEL = 1 * si.m
Z_MAX_PARCEL = 1000 * si.m
INITIAL_RH = 1

# this is here for testing functions from outside without command-line args
SIMULATION_TIMEOUT = None
LOG_FILENAME = None
PROCESS_NAME = os.getpid()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_filename", required=True)
    parser.add_argument("--sample_filename", required=True)
    parser.add_argument("--fail_filename", default=None)
    parser.add_argument("--save_period", default=5)
    parser.add_argument("--log_filename", default=None)
    parser.add_argument("--process_name", default=None)
    parser.add_argument("--simulation_timeout", default=None)

    args = parser.parse_args()

    LOG_FILENAME = args.log_filename
    PROCESS_NAME = (
        os.getpid()
        if args.process_name is None
        else f"{args.process_name} (PID {os.getpid()})"
    )
    SIMULATION_TIMEOUT = float(args.simulation_timeout)


def process_print(s):
    print(
        f"Process {PROCESS_NAME}: {s}",
        file=open(LOG_FILENAME, "a") if LOG_FILENAME else None,
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
                if "simulation_id" in prev_df.keys():
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


def run_simulation(simulation):
    if SIMULATION_TIMEOUT:
        return timeout_decorator.timeout(SIMULATION_TIMEOUT, use_signals=False)(
            simulation.run
        )()
    else:
        assert Z_MAX_PARCEL < np.inf
        return simulation.run()


# Run one parcel simulation, output the data for all timesteps
def generate_data_one_simulation(
    mode_Ns,
    mode_means,
    mode_stdevs,
    mode_kappas,
    velocity,
    initial_temperature,
    initial_pressure,
    n_sd=N_SD,
    dz_parcel=DZ_PARCEL,
    z_max_parcel=Z_MAX_PARCEL,
    initial_rh=INITIAL_RH,
    return_sim = False
):
    metadata = {
        k: v
        for d in (
            {
                f"mode_{i + 1}_N": mode_N,
                f"mode_{i + 1}_mean": mode_mean,
                f"mode_{i + 1}_stdev": mode_stdev,
                f"mode_{i + 1}_kappa": mode_kappa,
            }
            for i, (mode_N, mode_mean, mode_stdev, mode_kappa) in enumerate(
                zip(mode_Ns, mode_means, mode_stdevs, mode_kappas)
            )
        )
        for k, v in d.items()
    } | {
        "velocity": velocity,
        "initial_temperature": initial_temperature,
        "initial_pressure": initial_pressure,
    }

    dt = dz_parcel / velocity
    t_max = z_max_parcel / velocity

    settings = MyParcelSettings(
        mode_Ns=mode_Ns,
        mode_means=mode_means,
        mode_stdevs=mode_stdevs,
        mode_kappas=mode_kappas,
        velocity=velocity,
        initial_temperature=initial_temperature,
        initial_pressure=initial_pressure,
        dt=dt,
        t_max=t_max,
        n_sd_per_mode=n_sd // len(mode_Ns),
        initial_relative_humidity=initial_rh,
    )

    metadata["initial_vapor_mix_ratio"] = settings.initial_vapour_mixing_ratio

    success = True

    start_wall_time = time.time()

    simulation = None

    try:
        simulation = MyParcelSimulation(settings)
        metadata["solver"] = "PySDM"
        try:
            results = run_simulation(simulation)
        except RuntimeError as err:
            process_print(f"{err}: Retrying with SciPy condensation solver")
            simulation = MyParcelSimulation(
                settings,
                scipy_solver=True,
            )
            metadata["solver"] = "SciPy"
            results = run_simulation(simulation)
        products = results["products"]
        metadata["reached_z_max"] = results["reached_t_max"]
    except Exception as err:
        process_print(f"{type(err).__name__}: {err}")
        metadata["error"] = str(err).strip("\"'")
        products = simulation.get_output()
        success = False

    metadata["wall_time"] = time.time() - start_wall_time

    n_rows = len(list(products.values())[0])

    if n_rows > 0:
        saturation_mask = products["S_max"] >= 0
        if np.any(saturation_mask):
            metadata["saturation_temperature"] = products["T"][saturation_mask][0]
            metadata["saturation_pressure"] = products["p"][saturation_mask][0]
        else:
            metadata["saturation_temperature"] = np.nan
            metadata["saturation_pressure"] = np.nan
        result = pd.DataFrame(
            {k: [metadata[k]] * n_rows for k in metadata.keys()} | products
        )
    else:
        result = pd.DataFrame(metadata, index=[0])

    if return_sim:
        return result, success, simulation
    else:
        return result, success


def generate_data(
    parameters, out_filename, fail_filename, save_period, convert_units=True
):
    result_df = pd.DataFrame()

    num_simulations = len(parameters)

    for i in range(len(parameters)):
        process_print(f"Simulation {i + 1} / {num_simulations}")
        velocity, initial_temperature, initial_pressure = parameters[i][-3:]
        mode_Ns_or_NsMeans, mode_means, mode_stdevs, mode_kappas = (
            parameters[i][:-3].reshape(-1, 4).T
        )
        if convert_units:
            mode_NsMeans = 10**mode_Ns_or_NsMeans
            mode_means = 10**mode_means
            velocity = 10**velocity
            mode_Ns = mode_NsMeans / mode_means
        else:
            mode_Ns = mode_Ns_or_NsMeans
        simulation_df, success = generate_data_one_simulation(
            mode_Ns=mode_Ns,
            mode_means=mode_means,
            mode_stdevs=mode_stdevs,
            mode_kappas=mode_kappas,
            velocity=velocity,
            initial_temperature=initial_temperature,
            initial_pressure=initial_pressure,
        )
        if success:
            # Only take data at the time of maximum supersaturation
            simulation_df = simulation_df[
                simulation_df["S_max"] == np.max(simulation_df["S_max"])
            ].sample(1)
            simulation_df.insert(0, "simulation_id", i % save_period)
            result_df = pd.concat([result_df, simulation_df])
        elif fail_filename:
            save_data(simulation_df, fail_filename)
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
    try:
        main()
    except Exception as err:
        process_print(f"{type(err).__name__}: {err}")
        raise
