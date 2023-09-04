import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.PreprocessAerosolData
import CloudMicrophysics.ActivationEmulatorModels
import EnsembleKalmanProcesses as EKP
import StatsBase
import Random
using EnsembleKalmanProcesses.ParameterDistributions
import Plots as PL

rng = Random.MersenneTwister(1)

const FT = Float64

include("ReadAerosolDataset.jl")
include(joinpath(pkgdir(CM), "test", "create_parameters.jl"))
toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
default_param_set = cloud_microphysics_parameters(toml_dict)

X_train, Y_train = read_aerosol_dataset("datasets/1modal_dataset1_train.csv")

sample_size = 200
N_samples = 100
N_ensemble = 100
N_iterations_per_sample = 100

function G(X, parameters)
    (f_coeff_1_ARG2000, f_coeff_2_ARG2000, g_coeff_1_ARG2000, g_coeff_2_ARG2000, pow_1_ARG2000, pow_2_ARG2000) = FT.(parameters)
    cur_values = (;
        (
            name => getfield(default_param_set, name) for
            name in fieldnames(typeof(default_param_set))
        )...
    )
    overridden_values = merge(
        cur_values,
        (; f_coeff_1_ARG2000, f_coeff_2_ARG2000, g_coeff_1_ARG2000, g_coeff_2_ARG2000, pow_1_ARG2000, pow_2_ARG2000),
    )
    param_set = CMP.CloudMicrophysicsParameters(overridden_values...)
    return PreprocessAerosolData.get_ARG_act_frac(X, param_set)
end

function get_error_metrics(X, Y, ensemble)
    predictions = [G(X, ensemble[:, i]) for i in 1:N_ensemble]
    rmse = [sqrt.((p .- Y) .^ 2) for p in predictions]
    return [StatsBase.mean(d) for d in rmse]
end

prior = combine_distributions([
    constrained_gaussian("f_coeff_1_ARG2000", 0.5, 0.5, 0, Inf),
    constrained_gaussian("f_coeff_2_ARG2000", 2.5, 0.5, 0, Inf),
    constrained_gaussian("g_coeff_1_ARG2000", 1.0, 0.5, 0, Inf),
    constrained_gaussian("g_coeff_2_ARG2000", 0.25, 0.5, 0, Inf),
    constrained_gaussian("pow_1_ARG2000", 1.5, 0.5, 0, Inf),
    constrained_gaussian("pow_2_ARG2000", 0.75, 0.5, 0, Inf),
])

all_params_means = []
all_mean_error_metrics = []

for i in 1:N_samples
    println("Sample $(i)")

    inds = StatsBase.sample(
        1:DF.nrow(X_train),
        sample_size,
        replace = false,
    )
    X_train_sample = X_train[inds, :]
    Y_train_sample = Y_train[inds]

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

    ensemble_kalman_process = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        Y_train_sample,
        1.0 * EKP.I,
        EKP.Inversion(),
    )

    mean_error_metrics = []

    for j in 1:N_iterations_per_sample
        println(" Iteration $(j)")

        params_cur = EKP.get_ϕ_final(prior, ensemble_kalman_process)

        error_metrics = get_error_metrics(X_train_sample, Y_train_sample, params_cur)
        push!(mean_error_metrics, StatsBase.mean(error_metrics))

        params_mean_cur = EKP.get_ϕ_mean_final(prior, ensemble_kalman_process)
        println(params_mean_cur)

        G_ens = hcat([G(X_train_sample, params_cur[:, k]) for k in 1:N_ensemble]...)

        EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    end

    params_final = EKP.get_ϕ_final(prior, ensemble_kalman_process)

    error_metrics = get_error_metrics(X_train_sample, Y_train_sample, params_final)
    push!(mean_error_metrics, StatsBase.mean(error_metrics))

    params_mean_final = EKP.get_ϕ_mean_final(prior, ensemble_kalman_process)
    println(params_mean_final)

    push!(all_params_means, params_mean_final)
    push!(all_mean_error_metrics, mean_error_metrics)

end

plot = PL.plot(all_mean_error_metrics, legend = false, xlabel = "Iteration", ylabel = "Mean RMSE for Ensemble Members")
PL.savefig(plot, "plots/EKI_error_metrics.png")

final_params_means = StatsBase.mean(all_params_means)
println("Final params means:")
println(final_params_means)
