using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__)

Random.seed!(1234)

include("ps4_winkle_source.jl")

# Helper: small subsample for speed
df,X,Z,y = load_data(); N = min(200, size(X,1)); X = Matrix(X[1:N, :]); Z = Matrix(Z[1:N, :]); y = Vector(y[1:N]); 
K, J = size(X,2), length(unique(y)); 
@assert K==3 && size(Z,2)==8 "Unexpected X/Z shapes"

@testset "PS4 minimal but comprehensive tests" begin
    # --- Data sanity ---
    @test N>0 && length(y)==N && all(map(!ismissing, X)) && all(map(!ismissing, Z))
    @test all(issubset([minimum(y):maximum(y)...]), [y...])  # integer-coded

    # --- Gaussian-Legendre quadrature utilities ---
    nodes7, w7 = lgwt(7, -4, 4); φ = Normal()
    @test isapprox(sum(w7 .* pdf.(φ, nodes7)), 1.0; atol=1e-3)
    σ=2.0; n02=Normal(0,σ); nodes10, w10 = lgwt(10, -5σ, 5σ)
    @test isapprox(sum(w10 .* (nodes10.^2) .* pdf.(n02, nodes10)), σ^2; atol=1e-2)

    # --- Multinomial logit with alternative-specific Z ---
    # parameter length per source: K*(J-1) alphas + 1 gamma
    θ0 = [zeros(K*(J-1)); 0.0]
    ll0 = mlogit_with_Z(θ0, X, Z, y); @test isfinite(ll0) && ll0 ≥ 0
    θ1 = [0.1 .+ randn(K*(J-1)); 0.1]; @test isfinite(mlogit_with_Z(θ1, X, Z, y))
    # optimizer returns estimates and SEs with expected length
    θ̂, sê = optimize_mlogit(X, Z, y)
    @test length(θ̂)==K*(J-1)+1 && length(sê)==length(θ̂) && all(isfinite, θ̂) && all(x->x≥0 && isfinite(x), sê)
    # optimum beats zero vector
    @test mlogit_with_Z(θ̂, X, Z, y) ≤ ll0 + 1e-6

    # --- Mixed logit log-likelihoods: well-posed and consistent ---
    # shapes: K*(J-1) alphas + μ_γ + σ_γ
    θmix = [θ̂[1:end-1]; θ̂[end]; 0.25]  # collapse to logit when σ→0
    @test isfinite(mixed_logit_quad(θmix, X, Z, y, 5)) && isfinite(mixed_logit_mc(θmix, X, Z, y, 64))
    # σ→0 should approximate plain logit at γ = μ_γ
    θmix0 = [θ̂[1:end-1]; θ̂[end]; 1e-6]
    ll_quad = mixed_logit_quad(θmix0, X, Z, y, 7); ll_logit = mlogit_with_Z(θ̂, X, Z, y)
    @test isapprox(ll_quad, ll_logit; atol=1e-3, rtol=1e-3)

    # --- Optimizer wrappers (do not run full optim; they return startvals by design) ---
    s_quad = optimize_mixed_logit_quad(X, Z, y, 5); s_mc = optimize_mixed_logit_mc(X, Z, y)
    @test length(s_quad)==K*(J-1)+2 && length(s_mc)==K*(J-1)+2 && all(isfinite, s_quad) && all(isfinite, s_mc)

    # --- Monte Carlo practice: basic sanity if it returns (capture stdout otherwise) ---
    # just ensure the function is callable without error and prints something
    buf=IOBuffer(); redirect_stdout(buf) do; practice_quadrature(); variance_quadrature(); practice_monte_carlo(); end
    out = String(take!(buf)); @test occursin("Quadrature", out) && occursin("Monte Carlo", out)
end
