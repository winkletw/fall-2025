using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

# Set working directory
cd(@__DIR__)

include("ps8_winkle_source.jl")

main()

@testset "Problem Set 8 Tests" begin
    # Test data loading
    @testset "load_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        @test df isa DataFrame
        @test size(df, 2) >= 12  # Should have at least the core variables
        @test all(in(names(df)), [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr, :logwage])
        @test all(in(names(df)), [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])
    end

    # Test ASVAB correlations
    @testset "compute_asvab_correlations" begin
        # Create mock data
        n = 100
        mock_df = DataFrame(
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        cordf = compute_asvab_correlations(mock_df)
        @test cordf isa DataFrame
        @test size(cordf) == (6, 6)  # 6x6 correlation matrix
        @test all(diag(Matrix(cordf)) .≈ 1.0)  # Diagonal elements should be 1
    end

    # Test PCA generation
    @testset "generate_pca!" begin
        # Create mock data with some correlation structure
        n = 100
        ξ = randn(n)  # latent factor
        mock_df = DataFrame(
            asvabAR = 0.8 .* ξ .+ 0.2 .* randn(n),
            asvabCS = 0.7 .* ξ .+ 0.3 .* randn(n),
            asvabMK = 0.9 .* ξ .+ 0.1 .* randn(n),
            asvabNO = 0.6 .* ξ .+ 0.4 .* randn(n),
            asvabPC = 0.75 .* ξ .+ 0.25 .* randn(n),
            asvabWK = 0.85 .* ξ .+ 0.15 .* randn(n)
        )
        
        # Test PCA generation
        df_with_pca = generate_pca!(copy(mock_df))
        @test :asvabPCA in names(df_with_pca)
        @test length(df_with_pca.asvabPCA) == n
    end

    # Test Factor Analysis generation
    @testset "generate_factor!" begin
        # Use same mock data structure as PCA test
        n = 100
        ξ = randn(n)
        mock_df = DataFrame(
            asvabAR = 0.8 .* ξ .+ 0.2 .* randn(n),
            asvabCS = 0.7 .* ξ .+ 0.3 .* randn(n),
            asvabMK = 0.9 .* ξ .+ 0.1 .* randn(n),
            asvabNO = 0.6 .* ξ .+ 0.4 .* randn(n),
            asvabPC = 0.75 .* ξ .+ 0.25 .* randn(n),
            asvabWK = 0.85 .* ξ .+ 0.15 .* randn(n)
        )
        
        df_with_factor = generate_factor!(copy(mock_df))
        @test :asvabFactor in names(df_with_factor)
        @test length(df_with_factor.asvabFactor) == n
    end

    # Test matrix preparation for factor model
    @testset "prepare_factor_matrices" begin
        n = 100
        mock_df = DataFrame(
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:20, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            logwage = randn(n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        X, y, Xfac, asvabs = prepare_factor_matrices(mock_df)
        @test size(X, 2) == 7  # Including constant
        @test size(Xfac, 2) == 4  # Including constant
        @test size(asvabs, 2) == 6  # 6 ASVAB scores
        @test length(y) == n
        @test all(X[:, end] .== 1.0)  # Check constant term
        @test all(Xfac[:, end] .== 1.0)  # Check constant term
    end

    # Test factor model likelihood computation
    @testset "factor_model" begin
        # Create small mock dataset
        n = 50
        X = [ones(n) randn(n, 2)]  # 3 covariates including constant
        Xfac = [ones(n) randn(n)]  # 2 covariates including constant
        Meas = randn(n, 2)  # 2 measurements
        y = randn(n)
        
        # Create parameter vector (simplified model)
        L = size(Xfac, 2)  # covariates in measurement eqs
        J = size(Meas, 2)  # number of measurements
        K = size(X, 2)     # covariates in wage eq
        
        θ = vcat(
            vec(randn(L, J)),  # γ parameters
            randn(K),          # β parameters
            randn(J + 1),      # α parameters
            abs.(randn(J + 1)) # σ parameters (must be positive)
        )
        
        # Test likelihood computation
        ll = factor_model(θ, X, Xfac, Meas, y, 5)
        @test ll isa Real
        @test !isnan(ll)
        @test !isinf(ll)
    end
end 
