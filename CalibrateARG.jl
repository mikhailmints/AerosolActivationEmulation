import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.PreprocessAerosolData
import CloudMicrophysics.ActivationEmulatorModels
import EnsembleKalmanProcesses as EKP
import StatsBase
import Random
using EnsembleKalmanProcesses.ParameterDistributions
using Plots

rng = Random.MersenneTwister(1)

const FT = Float64

include("ReadAerosolDataset.jl")
include(joinpath(pkgdir(CM), "test", "create_parameters.jl"))
toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
default_param_set = cloud_microphysics_parameters(toml_dict)

X_train, Y_train = read_aerosol_dataset("datasets/1modal_dataset1_train.csv")
X_train = X_train[1:100, :]
Y_train = Y_train[1:100]

function G(parameters)
    (f_coeff_1_ARG2000, f_coeff_2_ARG2000, g_coeff_ARG2000) = FT.(parameters)
    cur_values = (;
        (
            name => getfield(default_param_set, name) for
            name in fieldnames(typeof(default_param_set))
        )...
    )
    overridden_values = merge(
        cur_values,
        (; f_coeff_1_ARG2000, f_coeff_2_ARG2000, g_coeff_ARG2000),
    )
    param_set = CMP.CloudMicrophysicsParameters(overridden_values...)
    return PreprocessAerosolData.get_ARG_act_frac(X_train, param_set)
end

prior = combine_distributions([
    constrained_gaussian("f_coeff_1_ARG2000", 0.5, 1, 0, Inf),
    constrained_gaussian("f_coeff_2_ARG2000", 2.5, 1, 0, Inf),
    constrained_gaussian("g_coeff_ARG2000", 0.25, 1, 0, Inf),
])

N_ensemble = 20
N_iterations = 20

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    Y_train,
    1.0 * EKP.I,
    EKP.Inversion(),
)

for i in 1:N_iterations
    params_i = EKP.get_ϕ_final(prior, ensemble_kalman_process)

    predictions = [G(params_i[:, i]) for i in 1:N_ensemble]

    abs_err = [abs.(p .- Y_train) for p in predictions]

    println([StatsBase.mean(d) for d in abs_err])

    G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = EKP.get_ϕ_final(prior, ensemble_kalman_process)

predictions = [G(final_ensemble[:, i]) for i in 1:N_ensemble]

abs_err = [abs.(p .- Y_train) for p in predictions]

println([StatsBase.mean(d) for d in abs_err])
