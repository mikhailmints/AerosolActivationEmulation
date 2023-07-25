from collections import namedtuple

AerosolData = namedtuple(
    "AerosolData",
    (
        "mode_N",
        "mode_mean",
        "mode_stdev",
        "mode_kappa",
        "velocity",
        "initial_temperature",
        "initial_pressure",
        "ARG_act_frac_CliMA",
    ),
)
