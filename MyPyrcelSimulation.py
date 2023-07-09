"""
Adapted from PySDM_examples.Pyrcel
"""

import numpy as np
from PySDM_examples.utils import BasicSimulation

import PySDM
from PySDM import Builder
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


class MyPyrcelSimulation:
    def __init__(
        self,
        settings,
        scipy_solver=False,
        rtol_thd=1e-10,
        rtol_x=1e-10,
        dt_cond_range=(1e-3 * si.second, 1 * si.second),
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
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
            environment=env,
            kappa_times_dry_volume=attributes["kappa times dry volume"],
        )
        attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)

        products = (
            PySDM_products.Time(name="time"),
            PySDM_products.AmbientRelativeHumidity(name="RH"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientPressure(name="p"),
            PySDM_products.AmbientWaterVapourMixingRatio(name="qv"),
            PySDM_products.WaterMixingRatio(name="qc"),
            PySDM_products.PeakSupersaturation(name="S_max", unit="%"),
            PySDM_products.ActivableFraction(name="act_frac"),
            PySDM_products.ActivatingRate(name="act_rate"),
        )

        self.particulator = builder.build(attributes=attributes, products=products)

        if scipy_solver:
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.output_attributes = {
            "volume": tuple([] for _ in range(self.particulator.n_sd))
        }
        self.settings = settings

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
            if k == "act_frac":
                continue
            value = v.get()
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]
            output[k].append(value)
        output["act_frac"].append(
            self.particulator.products["act_frac"].get(S_max=output["S_max"][-1])[0]
        )

    def run(self):
        output = {k: [] for k in self.particulator.products}
        self._save(output)
        for _ in range(
            0, self.settings.nt + 1, self.settings.steps_per_output_interval
        ):
            self.particulator.run(steps=self.settings.steps_per_output_interval)
            self._save(output)
            # if output["act_frac"][-1] == 1:
            #     break
        return {"products": output, "attributes": self.output_attributes}
