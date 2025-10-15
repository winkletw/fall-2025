using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)

include("ps6_winkle_source.jl")

main()

"""
Create a tiny synthetic wide-format dataset compatible with `load_and_reshape_data` expectations.
Returns: DataFrame with required columns.
"""
function _make_wide_df(; N::Int=2, T::Int=20)
    df = DataFrame()
    # Required id-level vars
    df.RouteUsage = rand(0.0:0.1:1.0, N)            # continuous-ish
    df.Branded    = rand(0:1, N)                    # 0/1
    df.Zst        = rand(1:2, N)                    # 1..zbin
    
    # Panel-wide columns
    for t in 1:T
        df[!, Symbol("Y$t")]    = rand(0:1, N)
        df[!, Symbol("Odo$t")]  = rand(0:50_000, N) # odometer-like
        df[!, Symbol("Xst$t")]  = rand(1:3, N)      # 1..xbin
    end
    return df
end

"""
Spin up a minimal local HTTP server that serves CSV text at `/data.csv`.
Returns: (task, server, url)
"""
function _serve_csv(csv_text::AbstractString; port::Int=9183)
    using HTTP, Sockets
    const _CSV_TEXT = csv_text
    router = HTTP.Router()
    HTTP.@register(router, "GET", "/data.csv") do req::HTTP.Request
        return HTTP.Response(200, _CSV_TEXT)
    end
    server_task = @async HTTP.serve(router, Sockets.localhost, port; verbose=false)
    url = "http://127.0.0.1:$(port)/data.csv"
    sleep(0.2) # give server a tick
    return (server_task, url)
end

"""
Build a long-format df compatible with `estimate_flexible_logit` with columns:
Y, Odometer, RouteUsage, Branded, time
"""
function _make_long_df(; N::Int=5, T::Int=6)
    Random.seed!(123)
    rows = N*T
    df = DataFrame(
        Y          = rand(0:1, rows),
        Odometer   = rand(10_000:10_000:100_000, rows),
        RouteUsage = rand(0.0:0.1:1.0, rows),
        Branded    = rand(0:1, rows),
        time       = repeat(1:T, inner=N)
    )
    return df
end

# -------------------------
# Tests
# -------------------------

@testset "load_and_reshape_data()" begin
    # Prepare a tiny CSV using the wide df and serve over local HTTP
    wide = _make_wide_df(N=3, T=20)
    io = IOBuffer()
    CSV.write(io, wide)
    csv_txt = String(take!(io))
    server_task, url = _serve_csv(csv_txt; port=9277)
    try
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        @test nrow(df_long) == 3*20
        @test all(in(names(df_long)).(Ref([:bus_id, :time, :Y, :Odometer, :Xstate, :Zst, :RouteUsage, :Branded])))
        @test size(Xstate) == (3, 20)
        @test length(Zstate) == 3
        @test length(Branded) == 3
        # Check time runs 1..20 for each bus
        @test all(byrow -> all(sort(unique(byrow.time)) .== collect(1:20)),
                  eachrow(combine(groupby(df_long, :bus_id), :time => identity)))
    finally
        # Shut down the server task
        try; Base.throwto(server_task, InterruptException()); catch; end
    end
end

@testset "estimate_flexible_logit()" begin
    df = _make_long_df(N=10, T=8)
    model = estimate_flexible_logit(df)
    @test model isa GLM.GeneralizedLinearModel
    # Prediction sanity
    p = GLM.predict(model, df)
    @test length(p) == nrow(df)
    @test all(0 .<= p .<= 1)
end

@testset "construct_state_space()" begin
    xval = [10_000.0, 20_000.0, 30_000.0]
    zval = [0.2, 0.8]
    xbin, zbin = length(xval), length(zval)
    xtran = zeros(xbin*zbin, xbin) # only size(xtran,1) is used by the function
    st = construct_state_space(xbin, zbin, xval, zval, xtran)
    @test nrow(st) == xbin*zbin
    # Odometer varies fastest every zbin groups
    @test st.Odometer == vcat(fill(xval[1], zbin), fill(xval[2], zbin), fill(xval[3], zbin))
    # RouteUsage repeats each z value for each x
    @test st.RouteUsage == repeat(zval, outer=xbin)
    @test all(st.Branded .== 0.0)
    @test all(st.time .== 0.0)
end

@testset "compute_future_values()" begin
    # Build a simple model to supply probabilities via predict()
    df = _make_long_df(N=12, T=6)
    flex = glm(@formula(Y ~ 1 + Odometer + RouteUsage + Branded + time),
               df, Binomial(), LogitLink())
    xval = [10_000.0, 30_000.0]
    zval = [0.1, 0.9]
    xbin, zbin = length(xval), length(zval)
    st = construct_state_space(xbin, zbin, xval, zval, zeros(xbin*zbin, xbin))
    T = 6
    β = 0.95
    FV = compute_future_values(st, flex, zeros(xbin*zbin, xbin), xbin, zbin, T, β)
    @test size(FV) == (xbin*zbin, 2, T+1)
    # t = 1 should remain 0 (loop is for t in 2:T)
    @test all(FV[:,:,1] .== 0.0)
    # Entries should be finite and nonnegative
    @test all(isfinite, FV)
    @test all(FV .>= 0.0)
end

@testset "compute_fvt1()" begin
    # Build minimal inputs
    N = 2; T = 20
    xbin = 3; zbin = 2
    # df_long only needs bus_id and time for this routine; provide scaffold
    df_long = DataFrame(bus_id=repeat(1:N, inner=T), time=repeat(1:T, N))
    # Xstate indices in 1..xbin
    Xstate = [1 for i in 1:N, t in 1:T]
    # Zstate indices in 1..zbin
    Zstate = [1, 2]
    # B vector (Branded) in {0,1}
    B = [0, 1]
    # xtran: make zero difference (row1-row0 == 0) so FVT1 => zeros
    xtran = zeros(xbin*zbin, xbin)
    # FV: any finite array with correct dims (values won't matter due to zero diff)
    FV = ones(xbin*zbin, 2, T+1)
    fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
    @test length(fvt1) == N*T
    @test all(fvt1 .== 0.0)
end

@testset "estimate_structural_params()" begin
    # Create a small long df with necessary columns
    df = _make_long_df(N=8, T=10)
    # Provide a random fvt1 vector with matching length
    fvt1 = randn(nrow(df))
    model = estimate_structural_params(df, fvt1)
    @test model isa GLM.GeneralizedLinearModel
    # Coefficients should exist for intercept, Odometer, Branded
    nm = coefnames(model)
    @test all(in(nm)).(Ref(["(Intercept)", "Odometer", "Branded"]))
end

@testset "API completeness" begin
    # Ensure main is defined (but don't run due to network dependency)
    @test isdefined(Main, :main)
end

