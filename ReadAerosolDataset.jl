import CSV
import DataFrames as DF
using DataFramesMeta

function get_num_modes(df::DataFrame)
    i = 1
    while true
        if !("mode_$(i)_N" in names(df))
            return i - 1
        end
        i += 1
    end
end

function read_aerosol_dataset(dataset_filename::String)
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
    Y = df.mode_1_act_frac_S_interp
    return (X, Y, df)
end
