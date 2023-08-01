import numpy as np
from PySDM.physics import si
import generate_parcel_data

generate_parcel_data.Z_MAX_PARCEL = 1000

NUM_PTS = 50

profiles = {
    "Silva_Ma_Fig7_mode_N": [
        [
            mode_N,
            0.05 * si.um,
            1.8,
            0.54,
            0.5 * si.m / si.s,
            283 * si.K,
            85000 * si.Pa,
        ]
        for mode_N in np.geomspace(1e1 / si.cm**3, 1e4 / si.cm**3, NUM_PTS)
    ],
    "Silva_Ma_Fig7_mode_mean": [
        [
            1000 / si.cm**3,
            mode_mean,
            1.8,
            0.54,
            0.5 * si.m / si.s,
            283 * si.K,
            85000 * si.Pa,
        ]
        for mode_mean in np.geomspace(1e-3 * si.um, 1e1 * si.um, NUM_PTS)
    ],
    "Silva_Ma_Fig7_velocity": [
        [
            1000 / si.cm**3,
            0.05 * si.um,
            1.8,
            0.54,
            velocity,
            283 * si.K,
            85000 * si.Pa,
        ]
        for velocity in np.geomspace(1e-2 * si.m / si.s, 1e1 * si.m / si.s, NUM_PTS)
    ],
    "Silva_Ma_Fig7_kappa": [
        [
            1000 / si.cm**3,
            0.05 * si.um,
            1.8,
            kappa,
            0.5 * si.m / si.s,
            283 * si.K,
            85000 * si.Pa,
        ]
        for kappa in np.linspace(0, 1.5, NUM_PTS)
    ],
}

for name, parameters in profiles.items():
    generate_parcel_data.generate_data(
        np.array(parameters),
        f"datasets/single_var_profiles/{name}.csv",
        None,
        5,
        convert_units=False,
    )
