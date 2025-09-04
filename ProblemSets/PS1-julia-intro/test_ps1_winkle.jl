using Test
using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, Distributions

cd(@__DIR__)
include("ps1_winkle.jl")

# matrixops tests---------------------------------------------------------------
@testset "matrixops()" begin
    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]
    out1, out2, out3 = matrixops(A, B)

    # i) element-wise product
    @test out1 == [5.0 12.0; 21.0 32.0]

    # ii) A' * B
    @test out2 == A' * B

    # iii) sum of all elements
    @test out3 == sum(A + B)

    # dimension mismatch must throw
    A_bad = rand(2,3); B_bad = rand(3,2)
    @test_throws ErrorException matrixops(A_bad, B_bad)
end

# q1 tests----------------------------------------------------------------------
@testset "q1()" begin
    # Run q1 and capture its returns
    Random.seed!(1234)  # keep deterministic
    A, B, C, D = q1()

    # Shapes
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)  # [A[1:5,1:5]  B[1:5,end-1:end]]
    @test size(D) == (10, 7)

    # A range check (Uniform(-5, 10))
    @test all((-5 .<= A) .& (A .<= 10))

    # D is A with positives zeroed out
    @test D == A .* (A .<= 0)

    # file outputs that q1() is expected to write (light integration check)
    # note: code writes "matrixC.csv", not "Cmatrix.csv".
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("matrixC.csv")
    @test isfile("Dmatrix.dat")

    # check on primary JLD contents
    saved = load("matrixpractice.jld")
    @test all(haskey(saved, k) for k in ["A","B","C","D","E","F","G"])
end

# q2 tests----------------------------------------------------------------------
@testset "q2(A,B,C)" begin
    # small, explicit inputs to exercise AB manual loop + filtering
    A = [1.0 -2.0; 3.5 0.0]
    B = [2.0  3.0; -1.0 4.0]
    C = [-6.0 -5.0; 0.0 5.0]  # includes values outside and inside [-5,5]

    # confirm function runs without error and returns nothing
    @test q2(A, B, C) === nothing

    # re-check AB loop logic equivalence (manual vs broadcasting)
    AB_expected = A .* B
    AB_manual = zeros(size(A))
    for r in axes(A,1), c in axes(A,2)
        AB_manual[r, c] = A[r, c] * B[r, c]
    end
    @test AB_manual ≈ AB_expected

    # re-check filtering logic equivalence for Cprime
    Cprime_expected = C[(C .>= -5) .& (C .<= 5)]
    Cprime_manual = Float64[]
    for c in axes(C, 2), r in axes(C, 1)
        if C[r, c] >= -5 && C[r, c] <= 5
            push!(Cprime_manual, C[r, c])
        end
    end
    @test Cprime_manual == Cprime_expected
end

# q2 β spec tests---------------------------------------------------------------
@testset "β spec checks (as in q2)" begin
    K, T = 6, 5
    β = zeros(K, T)
    β[1, :] = [1 + 0.25*(t - 1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t/3 for t in 1:T]

    @test size(β) == (K, T)
    @test β[1,1] == 1.0
    @test β[1,2] == 1.25
    @test β[2,1] == log(1)
    @test β[2,2] == log(2)
    @test β[3,1] == -sqrt(1)
    @test β[5,3] == 3
    @test β[6,4] == 4/3
    @test all(isfinite, β)
end

# q4 test-----------------------------------------------------------------------
@testset "q4() with hermetic JLD fixture" begin
    # provision a minimal fixture that satisfies q4 expectations
    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]
    C = [1.0 2.0; 3.0 4.0; 5.0 6.0]     # deliberately different size than D
    D = [1.0 2.0]
    E = rand(2,2); F = rand(2,2); G = rand(2,2)
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # run and return nothing
    @test q4() === nothing
end

# integration checks for q3/q4--------------------------------------------------
@testset "Integration (requires nlsw88.csv)" begin
    if isfile("nlsw88.csv")
        @test_nowarn q3()
        @test isfile("nlsw88_cleaned.csv")
    else
        @info "Skipping q3() integration test (nlsw88.csv not found)."
        @test true
    end
end

# tidy up test files------------------------------------------------------------
function cleanup_test_files()
    files_to_remove = [
        "matrixpractice.jld",
        "firstmatrix.jld",
        "matrixC.csv",
        "Dmatrix.dat",
        "nlsw88_cleaned.csv"
    ]
    for f in files_to_remove
        if isfile(f)
            try
                rm(f)
            catch e
                @warn "Failed to remove $(f)" exception=(e, catch_backtrace())
            end
        end
    end
end

# uncomment to clean after tests:
# cleanup_test_files()


#-------------------------------------------------------------------------------
# end of script
#-------------------------------------------------------------------------------