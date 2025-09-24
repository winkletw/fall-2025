using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("ps3_winkle_source.jl")

#-------------------------------------------------------------------------------
# Helpers to create toy datasets
#-------------------------------------------------------------------------------

"""
    make_toy_data(N::Int, J::Int; K::Int=3, seed::Int=123)

Create a small synthetic dataset:
- X: N x K matrix
- Z: N x J matrix (alt-specific covariates)
- y: Vector{Int} of length N with values in 1:J
"""
function make_toy_data(N::Int, J::Int; K::Int=3, seed::Int=123)
    Random.seed!(seed)
    X = randn(N, K)
    Z = randn(N, J)
    y = rand(1:J, N)
    return X, Z, y
end

# -------------------------------
# Tests for MNL likelihood only
# -------------------------------
@testset "PS3 Multinomial Logit (MNL) Workflow" begin
    @testset "Zero-parameter ⇒ uniform probabilities" begin
        N, J, K = 5, 4, 3
        X, Z, y = make_toy_data(N, J; K=K, seed=1)
        # theta = [alpha (K*(J-1) elements); gamma (1 element)]
        theta = zeros(K*(J-1) + 1)
        # Under theta=0, utilities are all 0 so P_j = 1/J for every obs
        # Negative log-likelihood should be N * log(J)
        nll = mlogit_with_Z(theta, X, Z, y)
        @test isapprox(nll, N * log(J); atol=1e-8)
    end

    @testset "Invariance to common shift in Z (because Z enters in differences)" begin
        N, J, K = 8, 5, 3
        X, Z, y = make_toy_data(N, J; K=K, seed=2)
        theta = vcat(randn(K*(J-1)), randn())  # random alphas + gamma
        c = 5.0
        # Add the same constant to every alternative's Z (including the base alt)
        Z_shift = Z .+ c
        nll1 = mlogit_with_Z(theta, X, Z, y)
        nll2 = mlogit_with_Z(theta, X, Z_shift, y)
        # Because the model uses (Z[:,j] - Z[:,end]) internally, this should be invariant
        @test isapprox(nll1, nll2; atol=1e-10)
    end

    @testset "Nearly deterministic choice with large gamma and favorable Z diff" begin
        # One observation, three alternatives. Base is the last alternative by construction in source.
        N, J, K = 1, 3, 3
        X = zeros(N, K)        # eliminate X effects
        Z = zeros(N, J)        # start with zeros so diffs are 0
        y = [2]                # the chosen alternative is j=2

        # Make alternative 2 much more attractive relative to the base (j=J)
        Z[1,2] = 1.0           # so (Z[:,2] - Z[:,J]) = 1.0
        gamma = 10.0           # large positive effect on that difference
        theta = vcat(zeros(K*(J-1)), gamma)  # zeros for alpha, big gamma

        nll = mlogit_with_Z(theta, X, Z, y)

        # Theoretical probability: P(y=2) = exp(gamma*1) / (exp(gamma*1) + exp(0) + exp(0))
        # Negative log-likelihood = -log(P(y=2))
        denom = exp(gamma*1) + 2.0
        target = -log(exp(gamma*1) / denom)
        @test isapprox(nll, target; atol=1e-8)
        @test nll < 1e-4  # should be extremely small
    end

    @testset "Throws with incorrect theta length" begin
        N, J, K = 4, 4, 3
        X, Z, y = make_toy_data(N, J; K=K, seed=3)
        # Wrong length: should be K*(J-1)+1 = 3*3+1 = 10
        theta_bad = zeros(7)
        @test_throws BoundsError mlogit_with_Z(theta_bad, X, Z, y)
    end
end


#-------------------------------------------------------------------------------
# Nested Logit (2-level)
#-------------------------------------------------------------------------------

# Utilities
# ---------
# Safe log-sum-exp for vectors
function _logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# Unpack alpha (K*(J-1)) into K×J with last column zeros (base alternative J)
function _unpack_alpha(theta::AbstractVector{<:Real}, K::Int, J::Int)
    expected = K*(J-1)
    @assert length(theta) >= expected "Parameter vector too short for alpha block"
    A = reshape(view(theta, 1:expected), K, J-1)
    Alpha = hcat(A, zeros(K))  # last column = base alt J
    return Alpha
end

# Parameter layout:
#   theta = [ vec(alpha[ :, 1:(J-1) ]) ; gamma ; lambda ]
# where size(alpha) = K × (J-1), gamma::Real, lambda::Real in (0,1]
function _unpack_params(theta::AbstractVector{<:Real}, K::Int, J::Int)
    expected_alpha = K*(J-1)
    @assert length(theta) == expected_alpha + 2 "theta should be length K*(J-1) + 2 (gamma, lambda)"
    Alpha = _unpack_alpha(theta, K, J)
    γ      = theta[expected_alpha + 1]
    λ      = theta[expected_alpha + 2]
    return Alpha, γ, λ
end

# Core: compute log-probabilities (per observation and alternative), given nests
# Returns:
#   logP :: Matrix{Float64} of size N×J
# Notes:
#   - Outside option contributes to the denominator only when include_outside=true.
function _nested_logit_logprobs(Alpha::AbstractMatrix{<:Real},
                                γ::Real, λ::Real,
                                X::AbstractMatrix{<:Real},
                                Z::AbstractMatrix{<:Real},
                                nests::Vector{Vector{Int}};
                                include_outside::Bool=true)

    @assert 0.0 < λ <= 1.0 "λ must lie in (0, 1]."
    N, K = size(X)
    N2, J = size(Z)
    @assert N == N2 "X and Z must have same number of rows (observations)."
    @assert size(Alpha, 1) == K && size(Alpha, 2) == J "Alpha must be K×J."
    @assert all(1 .<= vcat(nests...) .<= J) "Nest indices must be in 1:J."

    # Utilities: V_ij = X_i' α_j + γ * (Z_{ij} - Z_{iJ})
    # Precompute baseline column (J)
    ZJ = @view Z[:, J]
    V  = X * Alpha .+ γ .* (Z .- ZJ)  # N×J

    # For each obs i:
    #   logS_m(i) = log( ∑_{k ∈ m} exp( V_ik / λ ) )
    #   logNestTerm_m(i) = λ * logS_m(i)
    #   logDen(i) = log(  ∑_m exp(logNestTerm_m(i))  + outside )
    # where outside = 1 (if include_outside), contributing log(1)=0 in log-sum-exp.
    logP = similar(V, Float64)
    logS_per_nest = Vector{Float64}(undef, length(nests))
    logNestTerm   = similar(logS_per_nest)

    for i in 1:N
        # Compute per-nest aggregates
        for (m_idx, m) in enumerate(nests)
            # log ∑ exp(V/λ)
            logS_per_nest[m_idx] = _logsumexp(@view(V[i, m]) ./ λ)
            logNestTerm[m_idx]   = λ * logS_per_nest[m_idx]
        end

        if include_outside
            # Denominator includes 1 outside option (log-term = 0)
            logDen = _logsumexp(vcat(logNestTerm, 0.0))
        else
            logDen = _logsumexp(logNestTerm)
        end

        # Fill logP for each alternative, using its nest's aggregates
        for (m_idx, m) in enumerate(nests)
            ls = logS_per_nest[m_idx]            # logS_m
            for j in m
                # log P_ij = V_ij/λ + (λ - 1)*logS_m - logDen
                logP[i, j] = (V[i, j] / λ) + (λ - 1.0)*ls - logDen
            end
        end
    end

    return logP
end

# Negative log-likelihood for nested logit
# ----------------------------------------
# y is a Vector{Int} with values in 1:J (chosen alternative per obs)
function nested_logit_nll(theta::AbstractVector{<:Real},
                          X::AbstractMatrix{<:Real},
                          Z::AbstractMatrix{<:Real},
                          y::AbstractVector{<:Integer},
                          nests::Vector{Vector{Int}};
                          include_outside::Bool=true)

    N, K = size(X)
    J    = size(Z, 2)
    @assert length(y) == N "y length must match number of rows in X/Z."
    Alpha, γ, λ = _unpack_params(theta, K, J)
    logP = _nested_logit_logprobs(Alpha, γ, λ, X, Z, nests; include_outside=include_outside)

    # NLL = -∑ log P_{i, y_i}
    s = 0.0
    @inbounds for i in 1:N
        lp = logP[i, y[i]]
        @assert isfinite(lp) "Non-finite log-probability encountered."
        s -= lp
    end
    return s
end

# Predict probabilities
# ---------------------
# Returns an N×J matrix with probabilities for each alternative.
function predict_nested_logit_probabilities(theta::AbstractVector{<:Real},
                                            X::AbstractMatrix{<:Real},
                                            Z::AbstractMatrix{<:Real},
                                            nests::Vector{Vector{Int}};
                                            include_outside::Bool=true)
    N, K = size(X)
    J    = size(Z, 2)
    Alpha, γ, λ = _unpack_params(theta, K, J)
    logP = _nested_logit_logprobs(Alpha, γ, λ, X, Z, nests; include_outside=include_outside)
    return exp.(logP)
end

# Convenience optimizer wrapper (uses Optim.jl)
# ---------------------------------------------
# theta0 layout: [vec(alpha[:,1:(J-1)]); gamma; lambda]
# Tip: To keep λ in (0,1], you can parameterize λ = logistic(ρ) and optimize over ρ instead.
function optimize_nested_logit(theta0::AbstractVector{<:Real},
                               X::AbstractMatrix{<:Real},
                               Z::AbstractMatrix{<:Real},
                               y::AbstractVector{<:Integer},
                               nests::Vector{Vector{Int}};
                               include_outside::Bool=true,
                               optimizer = nothing,
                               maxiters::Int=1_000)

    @assert isdefined(Main, :Optim) "Load Optim.jl: using Optim"
    loss(θ) = nested_logit_nll(θ, X, Z, y, nests; include_outside=include_outside)

    if optimizer === nothing
        # Unconstrained optimizer; ensure your θ keeps λ in (0,1]
        # or wrap θ with a transform if you need strict constraints.
        res = Optim.optimize(loss, theta0, Optim.BFGS(); iterations=maxiters)
        return res
    else
        res = Optim.optimize(loss, theta0, optimizer; iterations=maxiters)
        return res
    end
end

# Example of packing parameters:
# -----------------------------
# function pack_theta(alpha::AbstractMatrix{<:Real}, gamma::Real, lambda::Real)
#     K, J = size(alpha)
#     @assert alpha[:, end] ≈ zeros(K) "Last column of alpha should be zeros for the base alternative."
#     return vcat(vec(alpha[:, 1:(J-1)]), gamma, lambda)
# end