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

function ARG_scheme(data_row::NamedTuple, num_modes)
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
    act_frac_per_mode =
        AA.N_activated_per_mode(
            param_set,
            ad,
            temperature,
            pressure,
            velocity,
            q,
        ) ./ mode_Ns
    max_supersaturation = AA.max_supersaturation(
        param_set,
        ad,
        temperature,
        pressure,
        velocity,
        q,
    )
    # critical_supersaturation =
    #     AA.critical_supersaturation(param_set, ad, temperature)
    # act_frac_per_mode_from_S_max =
    #     AA.N_activated_per_mode(
    #         param_set,
    #         ad,
    #         temperature,
    #         pressure,
    #         velocity,
    #         q,
    #         data_row.S_max,
    #         critical_supersaturation,
    #     ) ./ mode_Ns
    return (; act_frac_per_mode, max_supersaturation)
end

function read_aerosol_dataset(
    dataset_filename::String,
    keep_y_as_table::Bool = false,
)
    df = DF.DataFrame(CSV.File(dataset_filename))
    df = filter(row -> row.S_max > 0 && row.S_max < 0.2, df)
    selected_columns_X = []
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
    end
    append!(
        selected_columns_X,
        [:velocity, :initial_temperature, :initial_pressure],
    )
    X = df[:, selected_columns_X]
    if keep_y_as_table
        Y = DF.select(
            DF.transform(df, :S_max => ByRow(log) => :log_S_max),
            :log_S_max,
        )
    else
        Y = log.(df.S_max)
    end
    return (X, Y)
end

function preprocess_aerosol_data(X::DataFrame)
    num_modes = get_num_modes(X)
    X = DF.transform(
        X,
        AsTable(All()) =>
            ByRow(x -> log(ARG_scheme(x, num_modes).max_supersaturation)) =>
                :log_ARG_S_max,
    )
    for i in 1:num_modes
        X = DF.transform(
            X,
            Symbol("mode_$(i)_N") => ByRow(log) => Symbol("mode_$(i)_N"),
        )
        X = DF.transform(
            X,
            Symbol("mode_$(i)_mean") =>
                ByRow(log) => Symbol("mode_$(i)_mean"),
        )
    end
    X = DF.transform(X, :velocity => ByRow(log) => :velocity)
    return X
end
