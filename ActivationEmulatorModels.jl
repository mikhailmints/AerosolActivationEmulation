import MLJ
import MLJFlux
import Flux

Standardizer = MLJ.@load Standardizer pkg = MLJModels
EvoTreeRegressor = MLJ.@load EvoTreeRegressor pkg = EvoTrees
NeuralNetworkRegressor = MLJ.@load NeuralNetworkRegressor pkg = MLJFlux

function EvoTreeModel()
    model = EvoTreeRegressor()
    tuned_model = MLJ.TunedModel(
        tuning = MLJ.Grid(goal = 30),
        model = model,
        resampling = MLJ.CV(nfolds = 5),
        range = [
            range(model, :eta, lower = 0.05, upper = 1, scale = :log),
            range(model, :max_depth, lower = 3, upper = 15),
        ],
    )
    return tuned_model
end

struct MyFluxBuilder <: MLJFlux.Builder
    n1::Int
    n2::Int
    n3::Int
    dropout1::Float64
    dropout2::Float64
    dropout3::Float64
end

function MLJFlux.build(builder::MyFluxBuilder, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    return Flux.Chain(
        Flux.Dense(n_in => builder.n1, Flux.relu, init = init),
        Flux.Dropout(builder.dropout1),
        Flux.Dense(builder.n1 => builder.n2, Flux.relu, init = init),
        Flux.Dropout(builder.dropout2),
        Flux.Dense(builder.n2 => builder.n3, Flux.relu, init = init),
        Flux.Dropout(builder.dropout3),
        Flux.Dense(builder.n3 => n_out, init = init),
    )
end

function NNModel()
    model = NeuralNetworkRegressor(
        builder = MyFluxBuilder(50, 100, 30, 0.1, 0.3, 0.2),
        optimiser = Flux.Optimise.Adam(0.001, (0.9, 0.999), 1.0e-8),
        epochs = 200,
        loss = Flux.mse,
        batch_size = 50,
    )
    return model
end
