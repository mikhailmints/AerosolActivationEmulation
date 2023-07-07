import numpy as np
from PySDM import Builder, Formulae, products
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.environments import Parcel
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

DT = 1 * si.s
MASS_OF_DRY_AIR = 1e3 * si.kg
N_SD = 500

def get_activable_fraction(mode_N, mode_mean, mode_stdev, mode_kappa, velocity, qv, T, p, MAC):
    env = Parcel(dt=DT, mass_of_dry_air=MASS_OF_DRY_AIR, p0=p, q0=qv, T0=T, w=velocity)
    formulae = Formulae(constants={"MAC" : MAC})
    builder = Builder(n_sd=N_SD, backend=CPU(formulae))
    builder.set_environment(env)
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    spectrum = Lognormal(norm_factor=mode_N, m_mode=mode_mean, s_geom=mode_stdev)
    r_dry, n_per_volume = ConstantMultiplicity(spectrum).sample(N_SD)
    v_dry = formulae.trivia.volume(radius=r_dry)

