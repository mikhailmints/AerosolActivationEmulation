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

function ARG_activation_fraction(
    mode_N,
    mode_mean,
    mode_stdev,
    mode_kappa,
    velocity,
    temperature,
    pressure,
)
    ad = AM.AerosolDistribution((
        AM.Mode_κ(
            mode_mean,
            mode_stdev,
            mode_N,
            FT(1),
            FT(1),
            FT(0),
            mode_kappa,
            1,
        ),
    ))
    thermo_params = CMP.thermodynamics_params(param_set)
    pv0 = TD.saturation_vapor_pressure(thermo_params, temperature, TD.Liquid())
    vapor_mix_ratio =
        pv0 / TD.Parameters.molmass_ratio(thermo_params) / (pressure - pv0)
    q_vap = vapor_mix_ratio / (vapor_mix_ratio + 1)
    q = TD.PhasePartition(q_vap, FT(0), FT(0))
    N_act =
        AA.total_N_activated(param_set, ad, temperature, pressure, velocity, q)
    act_frac = N_act / mode_N
    return act_frac
end

function read_aerosol_dataset(dataset_filename::String, keep_y_as_table::Bool)
    df = DF.DataFrame(CSV.File(dataset_filename))
    X = @select(
        df,
        :mode_N,
        :mode_mean,
        :mode_stdev,
        :mode_kappa,
        :velocity,
        :initial_temperature,
        :initial_pressure,
    )
    Y = keep_y_as_table ? @select(df, :act_frac_S) : df.act_frac_S
    return (X, Y)
end



function preprocess_aerosol_data(X)
    X = @transform(
        X,
        :ARG_act_frac =
            ARG_activation_fraction.(
                :mode_N,
                :mode_mean,
                :mode_stdev,
                :mode_kappa,
                :velocity,
                :initial_temperature,
                :initial_pressure,
            )
    )
    X = @transform(
        X,
        :mode_N = log.(:mode_N),
        :mode_mean = log.(:mode_mean),
        :velocity = log.(:velocity)
    )
end
