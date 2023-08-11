"""
Adapted from PySDM_examples.Pyrcel
"""

from typing import Iterable

import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.initialisation.spectra import Lognormal


@strict
class MyParcelSettings:
    def __init__(
        self,
        mode_Ns: Iterable[float],
        mode_means: Iterable[float],
        mode_stdevs: Iterable[float],
        mode_kappas: Iterable[float],
        velocity: float,
        initial_temperature: float,
        initial_pressure: float,
        dt: float,
        t_max: float,
        n_sd_per_mode: float | Iterable[float],
        initial_relative_humidity: float = 1,
        formulae: Formulae = Formulae(),
    ):
        self.formulae = formulae
        self.n_sd_per_mode = (
            n_sd_per_mode
            if isinstance(n_sd_per_mode, Iterable)
            else (n_sd_per_mode,) * len(mode_Ns)
        )
        self.aerosol_modes_by_kappa = [
            (
                mode_kappa,
                Lognormal(norm_factor=mode_N, m_mode=mode_mean, s_geom=mode_stdev),
            )
            for (mode_N, mode_mean, mode_stdev, mode_kappa) in zip(
                mode_Ns, mode_means, mode_stdevs, mode_kappas
            )
        ]

        const = self.formulae.constants
        self.vertical_velocity = velocity
        self.initial_pressure = initial_pressure
        self.initial_temperature = initial_temperature
        pv0 = (
            initial_relative_humidity
            * formulae.saturation_vapour_pressure.pvs_Celsius(
                initial_temperature - const.T0
            )
        )
        self.initial_vapour_mixing_ratio = const.eps * pv0 / (initial_pressure - pv0)
        self.t_max = t_max
        self.timestep = dt
        self.output_interval = self.timestep

    @property
    def initial_air_density(self):
        const = self.formulae.constants
        dry_air_density = (
            self.formulae.trivia.p_d(
                self.initial_pressure, self.initial_vapour_mixing_ratio
            )
            / self.initial_temperature
            / const.Rd
        )
        return dry_air_density * (1 + self.initial_vapour_mixing_ratio)

    @property
    def nt(self) -> int:
        nt = self.t_max / self.timestep
        nt_int = round(nt)
        # np.testing.assert_almost_equal(nt, nt_int)
        return nt_int

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.timestep)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.nt + 1, self.steps_per_output_interval)
