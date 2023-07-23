from PySDM import Formulae
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si

from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver

from PySDM_examples.Pyrcel import Settings, Simulation


settings = Settings(
    dz = 1 * si.m,
    n_sd_per_mode = (1000,),
    aerosol_modes_by_kappa = {
        0.1885611585360898: Lognormal(
            norm_factor=40233789.92354 / si.m ** 3,
            m_mode=1.1669322087438114e-09 * si.m,
            s_geom=1.7322676373747328
        ),
    },
    vertical_velocity = 0.0143756504108188 * si.m / si.s,
    initial_pressure = 76309.09460546204 * si.Pa,
    initial_temperature = 296.50590490254416 * si.K,
    initial_relative_humidity = 1,
    displacement = 1000 * si.m,
    formulae = Formulae(constants={'MAC': 1})
)

simulation = Simulation(settings, products=[], rtol_thd=1e-2, rtol_x=1e-2, scipy_solver=True)

results = simulation.run()


