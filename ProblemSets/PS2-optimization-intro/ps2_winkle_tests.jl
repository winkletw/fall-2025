using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, GLM

cd(@__DIR__)

# Prevent side-effects if the source ends with `ps2()`
ENV["PS2_RUN"] = "0"

# Bring in the student's source (optional but harmless if not needed by tests)
try
    include("ps2_winkle_source.jl")
catch e
    @info "Could not include ps2_winkle_source.jl (tests still run): $e"
end

# ---- Local helpers (used if not provided at top-level by the source) ----
if !@isdefined(ols)
    function ols(beta, X, y)
        r = y .- X * beta
        return r' * r
    end
end

if !@isdefined(logit)
    function logit(alpha, X, y)
        # negative log-likelihood for binary logit
        p = 1 ./(1 .+ exp.(-X * alpha))
        return -sum((y .== 1) .* log.(p) .+ (y .== 0) .* log.(1 .- p))
    end
end

if !@isdefined(mlogit)
    function mlogit(alpha, X, y)
        # Multinomial logit negative log-likelihood with J classes.
        # Parameterization uses J-1 sets of K coefficients (last class normalized to zero).
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        for j in 1:J
            bigY[:, j] .= (y .== j)
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j in 1:J
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end
        P = num ./ repeat(dem, 1, J)
        return -sum(bigY .* log.(P))
    end
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Unified testset (Q1, Q2, Q3, Q4/5)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Problem Set 2 — Single-File Tests" begin
    # --------------------
    # Question 1: Optimization sanity
    # --------------------
    @testset "Q1: Polynomial optimizer" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        minusf(x) = -f(x)         # minimize minusf to maximize f
        Random.seed!(123)
        result = optimize(minusf, [-7.0], BFGS(); iterations=1_000, g_tol=1e-9)
        @test res.f_converged == true where res = result
        @test isapprox(Optim.minimizer(result)[1], -7.37824; atol=1e-4)
        @test isapprox(Optim.minimum(result), 964.313384; atol=1e-3)  # minimum of minusf = -maximum of f
    end

    # --------------------
    # Question 2: OLS
    # --------------------
    @testset "Q2: OLS — Optim ≈ closed form" begin
        Random.seed!(42)
        N, K = 600, 5
        X = [ones(N) randn(N, K-1)]
        β_true = [1.0, 2.0, -1.5, 0.75, 3.0]
        ϵ = 0.05 .* randn(N)
        y = X * β_true .+ ϵ

        β_cf = inv(X' * X) * X' * y
        β0 = randn(K)
        res = optimize(b -> ols(b, X, y), β0, BFGS(); iterations=800, g_tol=1e-8)
        β_hat = Optim.minimizer(res)

        @test res.f_converged
        @test isapprox(β_hat, β_cf; atol=1e-5, rtol=1e-5)
        @test isapprox(β_cf, β_true; atol=5e-2, rtol=5e-2)
    end

    # --------------------
    # Question 3: Binary Logit
    # --------------------
    @testset "Q3: Logit — Optim ≈ GLM" begin
        Random.seed!(7)
        N, K = 1_000, 4
        X = [ones(N) randn(N, K-1)]
        α_true = [0.3, -0.8, 1.2, 0.5]
        p = 1 ./(1 .+ exp.(-X * α_true))
        y = rand.(Bernoulli.(p))

        α0 = zeros(K)
        res = optimize(a -> logit(a, X, y), α0, BFGS(); iterations=1_200, g_tol=1e-7)
        α_opt = Optim.minimizer(res)

        df = DataFrame(y=y, x1=X[:,2], x2=X[:,3], x3=X[:,4])
        glm_fit = glm(@formula(y ~ x1 + x2 + x3), df, Binomial(), LogitLink())
        α_glm = coef(glm_fit)  # [Intercept, x1, x2, x3]

        @test res.f_converged
        @test isapprox(α_opt, α_glm; atol=5e-2, rtol=5e-2)
        @test sign.(α_opt) == sign.(α_glm)
    end

    # --------------------
    # Question 4/5: Multinomial Logit
    # --------------------
    @testset "Q4/5: Multinomial logit — recovery" begin
        Random.seed!(11)
        N, K, J = 1_000, 4, 3
        X = [ones(N) randn(N, K-1)]

        # True params: K × (J-1), last class normalized to 0
        A_true = [ 0.8   -0.4;
                   1.2    0.3;
                  -0.6    0.9;
                   0.5   -0.2 ]

        scores = X * A_true
        scores_full = [scores  zeros(N)]
        denom = sum(exp.(scores_full), dims=2)
        P = exp.(scores_full) ./ denom

        y = map(i -> argmax(rand(Multinomial(1, vec(P[i, :])))), 1:N)

        α_true_vec = vec(A_true)
        α0 = randn(length(α_true_vec))

        # Sanity: truth should beat random init in NLL
        @test mlogit(α_true_vec, X, y) < mlogit(α0, X, y)

        res = optimize(a -> mlogit(a, X, y), α0, BFGS(); iterations=2_000, g_tol=1e-7)
        α_hat = Optim.minimizer(res)

        @test res.f_converged
        @test size(α_hat) == size(α_true_vec)
        @test isapprox(α_hat, α_true_vec; atol=0.2, rtol=0.2)
        @test sign.(α_hat) == sign.(α_true_vec)
    end
end
