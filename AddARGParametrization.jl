import CloudMicrophysics as CM
import CloudMicrophysics: AerosolActivation as AA, AerosolModel as AM
import CLIMAParameters as CP
import Thermodynamics as TD
import CSV
import DataFrames as DF
using DataFramesMeta

const FT = Float64

include(joinpath(pkgdir(CM), "test", "create_parameters.jl"))
toml_dict = CP.create_toml_dict(Float64; dict_type = "alias")
param_set = cloud_microphysics_parameters(toml_dict)

function activation_fraction(
    mode_N,
    mode_mean,
    mode_stdev,
    mode_kappa,
    velocity,
    temperature,
    pressure,
    vapor_mix_ratio,
    water_mix_ratio,
)
    try
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
        q_vap = (vapor_mix_ratio) / (vapor_mix_ratio + 1)
        q_liq = (water_mix_ratio) / (water_mix_ratio + 1)
        q = TD.PhasePartition(q_vap + q_liq, q_liq, FT(0))
        N_act = AA.total_N_activated(
            param_set,
            ad,
            temperature,
            pressure,
            velocity,
            q,
        )
        act_frac = N_act / mode_N
        return act_frac
    catch
        return missing
    end
end

dataset_filename = joinpath("datasets", ARGS[1])

df = DF.DataFrame(CSV.File(dataset_filename))

if DF.columnindex(df, :ARG_act_frac) != 0
    DF.select!(df, Not(:ARG_act_frac))
end

df = @transform(
    DF.dropmissing(df),
    :ARG_act_frac =
        activation_fraction.(
            :mode_N,
            :mode_mean,
            :mode_stdev,
            :mode_kappa,
            :velocity,
            :initial_temperature,
            :initial_pressure,
            :vapor_mix_ratio,
            :water_mix_ratio,
        )
)

CSV.write(dataset_filename, df)
