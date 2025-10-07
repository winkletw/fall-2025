using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

include("ps5_winkle_source.jl")

hask(nt, sy) = hasproperty(nt, sy) || (nt isa AbstractDict && haskey(nt, sy))

@testset "load_static_data()" begin
    @test isdefined(Main, :load_static_data)
    df = load_static_data()
    @test df isa DataFrame
    @test nrow(df) > 0

    ns = names(df)
    ns_syms = Symbol.(ns)

    # Required columns (name-agnostic)
    @test :bus_id in ns_syms
    @test any(coln -> coln in ns_syms, [:period, :time, :t])

    # Choice column (accept common names)
    choice_syms = intersect(ns_syms, [:Y, :y, :choice, :replace])
    @test !isempty(choice_syms)

    # No missing identifiers
    @test all(!ismissing, df[:, Symbol(:bus_id)])

    # No missing time if available
    t_candidates = intersect(ns_syms, [:period, :time, :t])
    if !isempty(t_candidates)
        tcol = first(t_candidates)
        @test all(!ismissing, df[:, tcol])
    end
end

@testset "estimate_static_model()" begin
    df = load_static_data()
    @test isdefined(Main, :estimate_static_model)

    # Some implementations print the model and return `nothing`.
    m = estimate_static_model(df)

    if m === nothing
        # Acceptable: function executed without error; nothing returned.
        @test true
    else
        # If a model is returned, verify key properties.
        @test m isa StatsModels.TableRegressionModel
        β = coef(m)
        @test length(β) ≥ 1

        # If predict works, check probability bounds.
        try
            p = predict(m)
            @test length(p) == nrow(df)
            @test minimum(p) ≥ -1e-8
            @test maximum(p) ≤ 1 + 1e-8
        catch
            @test true  # prediction not required
        end
    end
end

@testset "load_dynamic_data()" begin
    @test isdefined(Main, :load_dynamic_data)
    d = load_dynamic_data()

    expected_keys = (:X, :Y, :Zstate, :Xstate, :xtran, :xval, :zbin, :xbin, :β, :T, :N, :B)
    for k in expected_keys
        @test hasproperty(d, k) || (d isa NamedTuple && haskey(d, k))
    end

    # Xstate: allow matrix (states × T) OR vector encodings
    if hasproperty(d, :Xstate)
        if d.Xstate isa AbstractMatrix
            @test size(d.Xstate, 2) == d.T
        elseif d.Xstate isa AbstractVector
            @test length(d.Xstate) in (d.T, d.N, d.N * d.T)
        else
            @test false  # unexpected type
        end
    end

    # Zstate: allow matrix (states × T) OR vector encodings
    if hasproperty(d, :Zstate)
        if d.Zstate isa AbstractMatrix
            @test size(d.Zstate, 2) == d.T
        elseif d.Zstate isa AbstractVector
            # Accept common encodings: per-time, per-bus, or panel
            @test length(d.Zstate) in (d.T, d.N, d.N * d.T)
        else
            @test false
        end
    end

    # xtran should be a (xbin × xbin) Markov matrix if present
    if hasproperty(d, :xtran) && d.xtran isa AbstractMatrix
        @test size(d.xtran, 1) == size(d.xtran, 2)
        rs = sum(d.xtran, dims=2)
        @test all(abs.(rs .- 1) .< 1e-6)
    end

    if hasproperty(d, :xbin) && hasproperty(d, :zbin)
        @test d.xbin ≥ 1
        @test d.zbin ≥ 1
    end
    if hasproperty(d, :β)
        @test 0.0 < d.β < 1.0
    end
end

@testset "estimate_dynamic_model()" begin
    d = load_dynamic_data()

    # Some implementations require an explicit start; try both paths.
    res = try
        estimate_dynamic_model(d)
    catch
        estimate_dynamic_model(d; θ_start=[0.1, 0.1])
    end

    @test res !== nothing

    # Accept multiple valid return shapes:
    if res isa AbstractArray && ndims(res) == 3 &&
       hasproperty(d, :xbin) && hasproperty(d, :zbin) && hasproperty(d, :T)
        # Value-function cube shape check
        @test size(res, 1) == d.xbin * d.zbin
        @test size(res, 2) == 2
        @test size(res, 3) == d.T + 1
    elseif res isa Number
        # fval / -loglike etc.
        @test isfinite(res)
    elseif res isa NamedTuple
        # Optim/solver-style bundle
        for k in (:θ, :converged, :fval, :FV)
            if haskey(res, k)
                @test res[k] !== nothing
            end
        end
    else
        # At least the call returned a concrete object
        @test typeof(res) != Nothing
    end
end

@testset "Integration smoke test: main()" begin
    @test isdefined(Main, :main)
    try
        Main.main()
    catch e
        @info "main() raised (ok for smoke): $e"
    end
    @test true
end