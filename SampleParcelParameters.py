import numpy as np
import argparse
import pickle
from scipy.stats import qmc


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


parser = argparse.ArgumentParser()
parser.add_argument("--num_simulations", required=True)
parser.add_argument("--num_processes", default=1)

args = parser.parse_args()

num_simulations = int(args.num_simulations)
num_processes = int(args.num_processes)

parameters = sample_parameters(num_simulations)

parameters_by_process = np.array_split(parameters, num_processes)

for i in range(num_processes):
    pickle.dump(parameters_by_process[i], open(f"samples/sample{i + 1}.pkl", "wb"))
