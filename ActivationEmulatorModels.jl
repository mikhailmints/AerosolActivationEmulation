import MLJ
import MLJFlux
import Flux
import GaussianProcesses
import StatsBase
import Optim
import QuasiMonteCarlo
import Distributions
import SpecialFunctions

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

# mutable struct BetaLik <: GaussianProcesses.Likelihood
#     p::Float64
#     q::Float64
#     priors::Array

#     BetaLik(p::Float64, q::Float64) = new(p, q, [])
# end

# function GaussianProcesses.log_dens(beta::BetaLik, f::AbstractVector, y::AbstractVector)
#     return -SpecialFunctions.logbeta(beta.p, beta.q) + (beta.p - 1) * log()
# end


mutable struct MyGPRegressor <: MLJ.Deterministic

end

function MLJ.fit(model::MyGPRegressor, verbosity, X, y)
    @info "Training GP model"
    gps = []
    for i in 1:1
        @info "Iteration $(i)"
        # weights = StatsBase.Weights([
        #     Distributions.pdf(Distributions.Normal(0.0, 0.5), x) for
        #     x in X.mode_1_ARG_act_frac
        # ])
        inds1 = StatsBase.sample(
            1:DF.nrow(X),
            #weights,
            1000,
            replace = false,
            ordered = true,
        )
        inds2 = StatsBase.sample(
            1:DF.nrow(X),
            #weights,
            50,
            replace = false,
            ordered = true,
        )
        gp = GaussianProcesses.SoR(
            Matrix(X)',
            Matrix(X[inds2, :])',
            y,
            GaussianProcesses.MeanZero(),
            GaussianProcesses.Mat52Ard(fill(2.0, DF.ncol(X)), 0.0),
            2.0,
        )
        GaussianProcesses.optimize!(gp)
        push!(gps, gp)
    end
    return gps, nothing, nothing
end

function MLJ.predict(::MyGPRegressor, fitresult, Xnew)
    return StatsBase.mean([
        GaussianProcesses.predict_f(gp, Matrix(Xnew)')[1] for gp in fitresult
    ])
end
