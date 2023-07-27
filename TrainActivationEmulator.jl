import MLJ

include("PreprocessAerosolData.jl")

include("ActivationEmulatorModels.jl")

X_train, Y_train = read_aerosol_dataset("datasets/dataset13_train.csv", true)

pipeline = preprocess_aerosol_data |> Standardizer() |> NNModel()

mach = MLJ.machine(pipeline, X_train, Y_train, 1 .- abs.(Y_train .- 0.5))

MLJ.fit!(mach, verbosity = 2)

