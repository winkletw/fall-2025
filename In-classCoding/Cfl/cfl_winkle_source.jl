

# =============================================================================
# 1. SETUP
# =============================================================================
function main_cfls()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
            df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    # Include likelihood function from PS4
    # include("../../ProblemSets/PS4-mixture/ps4_winkle_source.jl")

    function mlogit_with_Z(theta, X, Z, y)
        # extract parameters
        alpha = theta[1:end-1]  # first 21 elements
        gamma = theta[end]      # last element
        
        K = size(X, 2)  # number of covariates in X (3)
        J = length(unique(y))  # number of choices (8)
        N = length(y)   # number of observations
        
        # create choice indicator matrix
        bigY = zeros(N, J)
        for j = 1:J
            bigY[:, j] = y .== j
        end
        
        # reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
            
        # initialize probability matrix  
        T = promote_type(eltype(X), eltype(theta))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        
        # compute numerator for each choice j
        for j = 1:J
            num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
        end
        
        # compute denominator (sum of numerators)
        dem = sum(num, dims=2)
        
        # compute probabilities
        P = num ./ dem
        
        # compute negative log-likelihood
        loglike = -sum(bigY .* log.(P))
        
        return loglike
    end

    # Starting values
    θ_start = [.0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; 
            .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; 
            .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; 
            .1168824; -.2870554; -5.322248; 1.307477]

    # Estimate
    println("Estimating model...")
    td = TwiceDifferentiable(b -> mlogit_with_Z(b, X, Z, y), θ_start; autodiff = :forward)
    θ̂_optim = optimize(td, θ_start, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000))
    θ̂_mle = θ̂_optim.minimizer
    H = Optim.hessian!(td, θ̂_mle)

    println("Estimated parameters: ", round(θ̂_mle[end], digits=4))
    println("Wage coefficient (γ): ", round(θ̂_mle[end], digits=4))

    # compute model fit 
    J = length(unique(y))
    P = plogit(θ̂_mle, X, Z, J)

    # create comparison table 
    for j = 1:J
        println("Model fit for option ", j , ": (mean y) = ", mean(y.==j), 
                " (mean p) =", mean(P[:, j]))
    end 

    modelfit_df = DataFrame(occupation = 1:J,
                            data_pct = 100 * convert(Array, prop(freqtable(df, :occ_code))),
                            model_pct = 100 * vec(mean(P', dims=2)))
    
    modelfit_df.difference = modelfit_df.model_pct .- modelfit_df.data_pct

    println("\nModel Fit:")
    println(modelfit_df)
    println("\nMean Absolute Error: ", round(mean(abs.(modelfit_df.difference)), digits=4))

    # =============================================================================
    # 2. MODEL FIT
    # =============================================================================

    function plogit(θ, X, Z, J)
        # Fill in prediction function
        K = size(X, 2)
        N = size(X, 1)
        α = θ[1:end-1], θ
        γ = θ[end]
        
        # reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
        bigα = [reshape(α, K, J-1) zeros(K)]
            
        # initialize probability matrix  
        T = promote_type(eltype(X), eltype(θ))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        
        # compute numerator for each choice j
        for j = 1:J
            num[:,j] = exp.(X * bigα[:,j] .+ γ .* (Z[:,j] .- Z[:,J]))
        end
        
        # compute denominator (sum of numerators)
        dem = sum(num, dims=2)
        
        # compute probabilities
        P = num ./ dem
        
        return P
    end

    # Compute model fit
    J = length(unique(y))
    # P = plogit(θ̂_mle, X, Z, J)

    # Create comparison table
    # modelfit_df = DataFrame(...)

    

end