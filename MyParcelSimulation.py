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
        rtol_RH=1e-10,
        rtol_x=1e-10,
        dt_cond_range=(1e-8 * si.second, 1 * si.second),
        equilibrate=True,
        early_stop=True,
        max_iterations_without_increasing_smax=5,
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
            Condensation(rtol_RH=rtol_RH, rtol_x=rtol_x, dt_cond_range=dt_cond_range)
        )

        volume = env.mass_of_dry_air / settings.initial_air_density
        attributes = {
            k: np.empty(0) for k in ("dry volume", "kappa times dry volume", "n")
        }
        for i, (kappa, spectrum) in enumerate(settings.aerosol_modes_by_kappa.items()):
            sampling = ConstantMultiplicity(spectrum)
            r_dry, n_per_volume = sampling.sample(settings.n_sd_per_mode[i])
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes["n"] = np.append(attributes["n"], n_per_volume * volume)
            attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
            attributes["kappa times dry volume"] = np.append(
                attributes["kappa times dry volume"], v_dry * kappa
            )
        if equilibrate:
            r_wet = equilibrate_wet_radii(
                r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
                environment=env,
                kappa_times_dry_volume=attributes["kappa times dry volume"],
                rtol=1e-10,
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
            PySDM_products.ActivableFraction(name="act_frac_product"),
            PySDM_products.ActivatingRate(name="act_rate"),
        )

        self.particulator = builder.build(attributes=attributes, products=products)

        if scipy_solver:
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.output_attributes = {
            "n": tuple([] for _ in range(self.particulator.n_sd)),
            "volume": tuple([] for _ in range(self.particulator.n_sd)),
            "critical volume": tuple([] for _ in range(self.particulator.n_sd)),
            "critical supersaturation": tuple(
                [] for _ in range(self.particulator.n_sd)
            ),
        }
        self.settings = settings

        self.early_stop = early_stop
        self.max_iterations_without_increasing_smax = (
            max_iterations_without_increasing_smax
        )

        self.__sanity_checks(attributes, volume)

    def __sanity_checks(self, attributes, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.particulator.n_sd
        np.testing.assert_approx_equal(
            sum(attributes["n"]) / volume,
            sum(
                mode.norm_factor
                for mode in self.settings.aerosol_modes_by_kappa.values()
            ),
            significant=4,
        )

    def _save(self, output):
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        for k, v in self.particulator.products.items():
            if k == "act_frac_product":
                continue
            value = v.get()
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]
            output[k].append(value)
        if np.isnan(output["S_max"][-1]):
            output["S_max"][-1] = output["RH"][-1] - 1
        output["act_frac_product"].append(
            self.particulator.products["act_frac_product"].get(
                S_max=output["S_max"][-1] * 100
            )[0]
        )  # https://github.com/open-atmos/PySDM/issues/753
        act_num_S = 0
        act_num_V = 0
        max_RH = 1 + np.max(output["S_max"])
        volumes = np.array(self.output_attributes["volume"])[:, -1]
        radii = self.settings.formulae.trivia.radius(volumes)
        for drop_id in range(self.particulator.n_sd):
            if self.output_attributes["critical supersaturation"][drop_id][-1] < max_RH:
                act_num_S += self.output_attributes["n"][drop_id][-1]
            if (
                self.output_attributes["critical volume"][drop_id][-1]
                < volumes[drop_id]
            ):
                act_num_V += self.output_attributes["n"][drop_id][-1]
        total_multiplicity = np.sum(np.array(self.output_attributes["n"])[:, -1])
        output["act_frac_S"].append(act_num_S / total_multiplicity)
        output["act_frac_V"].append(act_num_V / total_multiplicity)
        output["mode_wet_radius_mean"].append(scipy.stats.gmean(radii))
        output["mode_wet_radius_stdev"].append(scipy.stats.gstd(radii))

    def run(self):
        output = {
            k: []
            for k in itertools.chain(
                self.particulator.products,
                (
                    "act_frac_S",
                    "act_frac_V",
                    "mode_wet_radius_mean",
                    "mode_wet_radius_stdev",
                ),
            )
        }
        while True:
            print(".", end="")
            self.particulator.run(steps=1)
            self._save(output)
            if output["time"][-1] > self.settings.t_max:
                break
            if (
                self.early_stop
                and np.max(output["S_max"]) > 0
                and output["S_max"].index(np.max(output["S_max"]))
                < len(output["S_max"]) - self.max_iterations_without_increasing_smax
            ):
                break
        output = {k: np.array(v) for k, v in output.items()}
        return {"products": output, "attributes": self.output_attributes}
