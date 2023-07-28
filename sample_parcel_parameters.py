import numpy as np
import argparse
import pickle
from scipy.stats import qmc
import os


def sample_parameters(num_simulations, num_modes):
    mode_param_bounds = [
        (1, 4),  # log10(mode_N), cm^-3
        (-3, 1),  # log10(mode_mean), um
        (1.5, 2),  # mode_stdev
        (0, 1.5),  # mode_kappa
    ]
    param_bounds = (mode_param_bounds * num_modes) + [
        (-2, 1),  # log10(velocity), m/s
        (248, 310),  # initial_temperature, K
        (50000, 105000),  # initial_pressure, Pa
    ]
    sampler = qmc.LatinHypercube(d=len(param_bounds))
    sample = sampler.random(n=num_simulations)
    l_bounds = [b[0] for b in param_bounds]
    u_bounds = [b[1] for b in param_bounds]
    result = qmc.scale(sample, l_bounds, u_bounds)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--num_simulations", required=True)
parser.add_argument("--num_modes", default=1)
parser.add_argument("--num_processes", default=1)

args = parser.parse_args()

out_dir = args.out_dir
num_simulations = int(args.num_simulations)
num_modes = int(args.num_modes)
num_processes = int(args.num_processes)

parameters = sample_parameters(num_simulations, num_modes)

parameters_by_process = np.array_split(parameters, num_processes)

for i in range(num_processes):
    pickle.dump(parameters_by_process[i], open(
        os.path.join(out_dir, f"sample{i + 1}.pkl"), "wb"))
