"""
Adapted from PySDM_examples.Pyrcel
"""

import numpy as np
import itertools
import scipy.stats

from PySDM import Builder
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import (
    ConstantMultiplicity,
    Logarithmic,
)
from PySDM.physics import si


class MyParcelSimulation:
    def __init__(
        self,
        settings,
        scipy_solver=False,
        scipy_rtol=1e-13,
        rtol_thd=1e-10,
        rtol_x=1e-10,
        rtol_equilibrate=1e-10,
        dt_cond_range=(1e-3 * si.second, 10 * si.second),
        equilibrate=True,
        early_stop=True,
        max_iterations_without_increasing_smax=5,
        console=False,
    ):
        env = Parcel(
            dt=settings.timestep,
            p0=settings.initial_pressure,
            q0=settings.initial_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.vertical_velocity,
            mass_of_dry_air=44 * si.kg,
        )
        n_sd = sum(settings.n_sd_per_mode)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(
            Condensation(rtol_thd=rtol_thd, rtol_x=rtol_x, dt_cond_range=dt_cond_range)
        )
        builder.request_attribute("critical supersaturation")

        volume = env.mass_of_dry_air / settings.initial_air_density
        attributes = {
            k: np.empty(0) for k in ("dry volume", "kappa times dry volume", "n")
        }
        self.mode_ids = np.empty(0, dtype=np.int32)
        self.num_modes = len(settings.aerosol_modes_by_kappa)
        for i, (kappa, spectrum) in enumerate(settings.aerosol_modes_by_kappa):
            sampling = Logarithmic(
                spectrum,
                size_range=spectrum.percentiles((0.01, 0.99)),
                error_threshold=0.05,
            )
            r_dry, n_per_volume = sampling.sample(settings.n_sd_per_mode[i])
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes["n"] = np.append(attributes["n"], n_per_volume * volume)
            attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
            attributes["kappa times dry volume"] = np.append(
                attributes["kappa times dry volume"], v_dry * kappa
            )
            self.mode_ids = np.append(
                self.mode_ids, np.full(settings.n_sd_per_mode[i], i, dtype=np.int32)
            )
        if equilibrate:
            r_wet = equilibrate_wet_radii(
                r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
                environment=env,
                kappa_times_dry_volume=attributes["kappa times dry volume"],
                rtol=rtol_equilibrate,
            )
            attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)
        else:
            attributes["volume"] = attributes["dry volume"].copy()

        products = (
            PySDM_products.Time(name="time"),
            PySDM_products.AmbientRelativeHumidity(name="RH"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientPressure(name="p"),
            PySDM_products.AmbientWaterVapourMixingRatio(
                name="vapor_mix_ratio", var="qv"
            ),
            PySDM_products.WaterMixingRatio(name="water_mix_ratio"),
            PySDM_products.PeakSupersaturation(name="S_max"),
        )

        self.particulator = builder.build(attributes=attributes, products=products)

        if scipy_solver:
            scipy_ode_condensation_solver.rtol = scipy_rtol
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.output_attributes = {
            "n": tuple([] for _ in range(self.particulator.n_sd)),
            "volume": tuple([] for _ in range(self.particulator.n_sd)),
            "critical volume": tuple([] for _ in range(self.particulator.n_sd)),
            "critical supersaturation": tuple(
                [] for _ in range(self.particulator.n_sd)
            ),
        }

        self.output = {
            k: []
            for k in itertools.chain(
                self.particulator.products,
                *(
                    (
                        f"mode_{i + 1}_act_frac_S",
                        f"mode_{i + 1}_act_frac_S_interp",
                        f"mode_{i + 1}_act_frac_V",
                        f"mode_{i + 1}_wet_radius_mean",
                        f"mode_{i + 1}_wet_radius_stdev",
                    )
                    for i in range(self.num_modes)
                ),
            )
        }

        self.settings = settings
        self.early_stop = early_stop
        self.max_iterations_without_increasing_smax = (
            max_iterations_without_increasing_smax
        )
        self.console = console

        self.__sanity_checks(attributes, volume)

    def __sanity_checks(self, attributes, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.particulator.n_sd

    def get_output(self):
        return {k: np.array(v) for k, v in self.output.items()}

    def _save(self):
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        for k, v in self.particulator.products.items():
            value = v.get()
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]
            self.output[k].append(value)
        if np.isnan(self.output["S_max"][-1]):
            self.output["S_max"][-1] = self.output["RH"][-1] - 1
        act_num_S = np.zeros(self.num_modes)
        act_num_V = np.zeros(self.num_modes)
        total_multiplicity = np.zeros(self.num_modes)
        max_RH = 1 + np.max(self.output["S_max"])
        volumes = np.array(
            [
                np.array(self.output_attributes["volume"])[self.mode_ids == i, -1]
                for i in range(self.num_modes)
            ]
        )
        radii = self.settings.formulae.trivia.radius(volumes)
        highest_critsat_activated = [None] * self.num_modes
        highest_critsat_activated_id = [None] * self.num_modes
        lowest_critsat_unactivated = [None] * self.num_modes
        lowest_critsat_unactivated_id = [None] * self.num_modes
        get_critsat = lambda drop_id: self.output_attributes[
            "critical supersaturation"
        ][drop_id][-1]
        for drop_id in range(self.particulator.n_sd):
            mode_id = self.mode_ids[drop_id]
            multiplicity = self.output_attributes["n"][drop_id][-1]
            total_multiplicity[mode_id] += multiplicity
            if get_critsat(drop_id) < max_RH:
                if (
                    highest_critsat_activated_id[mode_id] is None
                    or get_critsat(drop_id) > highest_critsat_activated[mode_id]
                ):
                    highest_critsat_activated[mode_id] = get_critsat(drop_id)
                    highest_critsat_activated_id[mode_id] = drop_id
                act_num_S[mode_id] += multiplicity
            elif (
                lowest_critsat_unactivated_id[mode_id] is None
                or get_critsat(drop_id) < lowest_critsat_unactivated[mode_id]
            ):
                lowest_critsat_unactivated[mode_id] = get_critsat(drop_id)
                lowest_critsat_unactivated_id[mode_id] = drop_id
            if (
                self.output_attributes["critical volume"][drop_id][-1]
                < self.output_attributes["volume"][drop_id][-1]
            ):
                act_num_V[mode_id] += multiplicity
        # Interpolate activated fraction between highest critical supersaturation
        # droplet activated and lowest critical supersaturation droplet unactivated
        act_num_S_interp = [
            (
                act_num_S[i]
                + (
                    (max_RH - (highest_critsat_activated[i]))
                    / (lowest_critsat_unactivated[i] - highest_critsat_activated[i])
                )
                * self.output_attributes["n"][lowest_critsat_unactivated_id[i]][-1]
            )
            if highest_critsat_activated_id[i] and lowest_critsat_unactivated_id[i]
            else act_num_S[i]
            for i in range(self.num_modes)
        ]
        for i in range(self.num_modes):
            self.output[f"mode_{i + 1}_act_frac_S"].append(
                act_num_S[i] / total_multiplicity[i]
            )
            self.output[f"mode_{i + 1}_act_frac_S_interp"].append(
                act_num_S_interp[i] / total_multiplicity[i]
            )
            self.output[f"mode_{i + 1}_act_frac_V"].append(
                act_num_V[i] / total_multiplicity[i]
            )
            self.output[f"mode_{i + 1}_wet_radius_mean"].append(
                scipy.stats.gmean(radii[i])
            )
            self.output[f"mode_{i + 1}_wet_radius_stdev"].append(
                scipy.stats.gstd(radii[i])
            )

    def run(self):
        reached_t_max = False
        while True:
            self.particulator.run(steps=1)
            self._save()
            if self.console:
                print(f"S_max: {self.output['S_max'][-1]}")
            if self.output["time"][-1] > self.settings.t_max:
                reached_t_max = True
                break
            if self.early_stop:
                # Went past supersaturation peak
                if (
                    np.max(self.output["S_max"]) > 0
                    and np.argmax(self.output["S_max"])
                    < len(self.output["S_max"])
                    - self.max_iterations_without_increasing_smax
                ):
                    break
                # Everything was activated
                if all(
                    [
                        self.output[f"mode_{i + 1}_act_frac_S"] == 1
                        for i in range(self.num_modes)
                    ]
                ):
                    break
        return {
            "products": self.get_output(),
            "attributes": self.output_attributes,
            "reached_t_max": reached_t_max,
        }
