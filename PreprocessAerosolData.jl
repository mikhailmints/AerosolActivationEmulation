import CloudMicrophysics as CM
import CloudMicrophysics:
    AerosolActivation as AA, AerosolModel as AM, Parameters as CMP
import CLIMAParameters as CP
import Thermodynamics as TD
import CSV
import DataFrames as DF
using DataFramesMeta

const FT = Float64

include(joinpath(pkgdir(CM), "test", "create_parameters.jl"))
toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
param_set = cloud_microphysics_parameters(toml_dict)

function get_num_modes(df::DataFrame)
    i = 1
    while true
        if !("mode_$(i)_N" in names(df))
            return i - 1
        end
        i += 1
    end
end

function ARG_activation_fraction(data_row::NamedTuple, num_modes)
    mode_Ns = []
    mode_means = []
    mode_stdevs = []
    mode_kappas = []
    velocity = data_row.velocity
    temperature = data_row.initial_temperature
    pressure = data_row.initial_pressure
    for i in 1:num_modes
        push!(mode_Ns, data_row[Symbol("mode_$(i)_N")])
        push!(mode_means, data_row[Symbol("mode_$(i)_mean")])
        push!(mode_stdevs, data_row[Symbol("mode_$(i)_stdev")])
        push!(mode_kappas, data_row[Symbol("mode_$(i)_kappa")])
    end
    ad = AM.AerosolDistribution(
        Tuple(
            AM.Mode_κ(
                mode_means[i],
                mode_stdevs[i],
                mode_Ns[i],
                FT(1),
                FT(1),
                FT(0),
                mode_kappas[i],
                1,
            ) for i in 1:num_modes
        ),
    )
    thermo_params = CMP.thermodynamics_params(param_set)
    pv0 = TD.saturation_vapor_pressure(thermo_params, temperature, TD.Liquid())
    vapor_mix_ratio =
        pv0 / TD.Parameters.molmass_ratio(thermo_params) / (pressure - pv0)
    q_vap = vapor_mix_ratio / (vapor_mix_ratio + 1)
    q = TD.PhasePartition(q_vap, FT(0), FT(0))
    N_act_per_mode = AA.N_activated_per_mode(
        param_set,
        ad,
        temperature,
        pressure,
        velocity,
        q,
    )
    act_frac_per_mode = N_act_per_mode ./ mode_Ns
    return act_frac_per_mode
end

function read_aerosol_dataset(dataset_filename::String, keep_y_as_table::Bool)
    df = DF.DataFrame(CSV.File(dataset_filename))
    selected_columns_X = []
    selected_columns_Y = []
    num_modes = get_num_modes(df)
    for i in 1:num_modes
        append!(
            selected_columns_X,
            Symbol.([
                "mode_$(i)_N",
                "mode_$(i)_mean",
                "mode_$(i)_stdev",
                "mode_$(i)_kappa",
            ]),
        )
        push!(selected_columns_Y, Symbol("mode_$(i)_act_frac_S"))
    end
    append!(
        selected_columns_X,
        [:velocity, :initial_temperature, :initial_pressure],
    )
    X = df[:, selected_columns_X]
    if keep_y_as_table
        Y = df[:, selected_columns_Y]
    else
        @assert size(selected_columns_Y) == 1
        Y = df.mode_1_act_frac_S
    end
    return (X, Y)
end

function preprocess_aerosol_data(X::DataFrame)
    num_modes = get_num_modes(X)
    X = DF.transform(
        X,
        AsTable(All()) =>
            ByRow(x -> ARG_activation_fraction(x, num_modes))
            => [Symbol("mode_$(i)_ARG_act_frac") for i in 1:num_modes],
    )
    for i in 1:num_modes
        X = DF.transform(
            X,
            Symbol("mode_$(i)_N") => ByRow(log) => Symbol("mode_$(i)_N"),
        )
        X = DF.transform(
            X,
            Symbol("mode_$(i)_mean") => ByRow(log) => Symbol("mode_$(i)_mean"),
        )
    end
    X = DF.transform(X, :velocity => ByRow(log) => :velocity)
    return X
end
