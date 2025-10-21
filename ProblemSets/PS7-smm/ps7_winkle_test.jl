using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("ps7_winkle_source.jl")

main()

# ------------------------- Unit tests -------------------------------------


# Helpers: create a tiny synthetic dataset to test functions deterministically
function tiny_wage_df()
	# create a small DataFrame-like object compatible with code that expects
	# columns: age, race, collgrad, wage, occupation
	using DataFrames
	df = DataFrame(age = [25, 30, 45, 22], race = [1, 2, 1, 2], collgrad = [1,0,1,0], wage = [10.0, 12.0, 9.0, 11.0], occupation = [1,2,3,4])
	return df
end

@testset "ps7_winkle_source tests" begin
	# Test load_data with a small in-memory CSV (avoid network)
	@testset "load_data" begin
		using CSV, Tempfile
		df = tiny_wage_df()
		# write to a temp file and use file URL path
		tmp = tempname()
		CSV.write(tmp, df)
		# CSV.read accepts a filename; to reuse load_data signature which expects
		# HTTP.get(url).body we call CSV.read directly here to validate parsing
		df2, X, y = load_data(tmp)  # load_data accepts a URL in original code; our test will pass the path
		@test size(X, 1) == nrow(df2)
		@test size(X, 2) == 4
		@test length(y) == nrow(df2)
		# spot check log wage
		@test isapprox(y[1], log(df2.wage[1]))
	end

	@testset "prepare_occupation_data" begin
		df = tiny_wage_df()
		df.occupation[4] = 8  # trigger collapsing into 7 in source code
		df2, Xocc, yocc = prepare_occupation_data(df)
		@test all(ismissing.(df2.occupation) .== false)
		@test size(Xocc, 2) == 4
		@test all(yocc .>= 1)
	end

	@testset "ols_gmm" begin
		# Build a simple linear model where true beta = [1, 2]
		X = [ones(5)  collect(1:5)]
		β_true = [1.0, 2.0]
		y = X * β_true
		# ols_gmm should be minimized at β_true and objective zero there
		J_at_true = ols_gmm(β_true, X, y)
		@test isapprox(J_at_true, 0.0, atol=1e-12)
		# small perturbation increases objective
		J_pert = ols_gmm(β_true .+ [0.1, -0.1], X, y)
		@test J_pert > J_at_true
	end

	@testset "mlogit_mle and probability checks" begin
		# Create tiny X and β to produce deterministic probabilities
		X = [ones(6)  repeat([0.0,1.0,0.0], inner=2) ]  # 6x2 small design
		# Build artificial y with 3 choices
		J = 3
		N = size(X,1)
		# Create α vector corresponding to zeros (so equal utilities)
		α = zeros(size(X,2)*(J-1))
		# With α = 0, probabilities should be uniform across J
		ll = mlogit_mle(α, X, vcat(fill(1,2), fill(2,2), fill(3,2)))
		@test isfinite(ll)
		# compute implied probabilities via internal reshaping logic by calling mlogit_mle
		# small sanity: negative log-likelihood should be >= 0
		@test ll >= 0
	end

	@testset "mlogit_gmm moments" begin
		# Use a small simulated dataset from sim_logit_w_gumbel
		Y, Xsim = sim_logit_w_gumbel(200, 3)
		# random α (zeros) should yield finite objective
		α0 = zeros(size(Xsim,2)*(3-1))
		Jval = mlogit_gmm(α0, Xsim, Y)
		@test isfinite(Jval)
		# Over-identified version should also be finite
		Jval_over = mlogit_gmm_overid(α0, Xsim, Y)
		@test isfinite(Jval_over)
	end

	@testset "sim_logit properties" begin
		Y, Xsim = sim_logit(1000, 4)
		@test length(Y) == 1000
		@test size(Xsim,1) == 1000
		@test minimum(Y) >= 1 && maximum(Y) <= 4
		# Check that empirical frequencies sum to ~1 across categories
		freqs = [mean(Y .== j) for j in 1:4]
		@test isapprox(sum(freqs), 1.0; atol=1e-8)
	end

	@testset "mlogit_smm_overid small D" begin
		# Use small data to keep test fast
		Y, Xsim = sim_logit(200, 3)
		α0 = zeros(size(Xsim,2)*(3-1))
		# D small to keep runtime low; function should return finite objective
		J_smm = mlogit_smm_overid(α0, Xsim, Y, 5)
		@test isfinite(J_smm)
	end
end


