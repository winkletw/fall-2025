#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return as a DataFrame.

# Arguments
- `url::String`: URL to the CSV file

# Returns
- DataFrame containing the NLSY data
"""
function load_data(url::String)
    return CSV.read(HTTP.get(url).body, DataFrame)    
end

"""
    estimate_base_regression(df::DataFrame)

Estimate the baseline wage regression without ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()
"""
function estimate_base_regression(df::DataFrame)
    # Use the @formula macro and lm() function to estimate the regression
    # Formula: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
end


#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Compute the correlation matrix for the six ASVAB test scores.

# Arguments
- `df::DataFrame`: Data containing ASVAB scores in the last 6 columns

# Returns
- DataFrame with correlation matrix (6x6)

# Hint
- The ASVAB scores are in columns: asvabAR, asvabCS, asvabMK, asvabNO, asvabPC, asvabWK
- These are the last 6 columns of the DataFrame
- Use cor() function on the matrix of ASVAB scores
"""
function compute_asvab_correlations(df::DataFrame)
    # Extract the last 6 columns as a matrix
    asvabs = Matrix(df[:, end-5:end])

    # Compute correlation matrix using cor()
    correlation = cor(asvabs)

    # Return as a formatted DataFrame
    cordf = DataFrame(
        cor1 = correlation[1, :],
        cor2 = correlation[2, :],
        cor3 = correlation[3, :],
        cor4 = correlation[4, :],
        cor5 = correlation[5, :],
        cor6 = correlation[6, :]
    )
    return cordf
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Estimate wage regression including all six ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()

# Question to consider:
- Given the correlations you computed, what problems might arise?
"""
function estimate_full_regression(df::DataFrame)
    # Estimate regression with all ASVAB scores included
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    generate_pca!(df::DataFrame)

Perform PCA on ASVAB scores and include first principal component in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first principal component

# Key steps:
1. Extract ASVAB scores as a matrix
2. IMPORTANT: Transpose to J×N (MultivariateStats requires features × observations)
3. Fit PCA model with maxoutdim=1
4. Transform the data to get principal component scores
5. Add PC scores to DataFrame and run regression

# Hints:
- asvabMat = Matrix(df[:, end-5:end])'  (note the transpose!)
- M = fit(PCA, asvabMat; maxoutdim=1)
- asvabPCA = MultivariateStats.transform(M, asvabMat)
- asvabPCA will be 1×N, need to reshape for regression
"""
function generate_pca!(df::DataFrame) # ! = modify in place
    # Extract ASVAB matrix and transpose to J×N
    asvabs = Matrix(df[:, end-5:end])' # take transpose
    
    # Fit PCA model with maxoutdim=1 
    M = fit(PCA, asvabs; maxoutdim=1)
    
    # Transform data to get principal component
    asvabPCA = MultivariateStats.transform(M, asvabs)
    
    # Add PC to dataframe (careful with dimensions!)
    df = @transform(df, :asvabPCA = asvabPCA[:])
    return df    
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    generate_factor!(df::DataFrame)

Perform Factor Analysis on ASVAB scores and include first factor in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first factor

# Note:
- Syntax is nearly identical to PCA, just use FactorAnalysis instead
"""
function generate_factor!(df::DataFrame) # ! = modify in place
    # Extract ASVAB matrix and transpose to J×N
    asvabs = Matrix(df[:, end-5:end])' # take transpose
    
    # Fit Factor model with maxoutdim=1 
    M = fit(FactorAnalysis, asvabs; maxoutdim=1)
    
    # Transform data to get factors
    asvabFactor = MultivariateStats.transform(M, asvabs)
    
    # Add Factor to dataframe (careful with dimensions!)
    df = @transform(df, :asvabFactor = asvabFactor[:])
    return df    
end


#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#
include("lgwt.jl")  # Include Gauss-Legendre quadrature function

"""
    prepare_factor_matrices(df::DataFrame)

Prepare the data matrices needed for the factor model estimation.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- `X`: Covariates for wage equation (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: Log wage outcomes (N×1)
- `Xfac`: Covariates for measurement equations (N×4: black, hispanic, female, constant)
- `asvabs`: Matrix of all 6 ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    # Create X matrix for wage equation (include constant at end)
    X = [df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr ones(size(df, 1))]
    y = df.logwage
    Xfac = [df.black df.hispanic df.female ones(size(df, 1))]
    asvabs = [df.asvabAR df.asvabCS df.asvabMK df.asvabNO df.asvabPC df.asvabWK]
    return X, y, Xfac, asvabs
end

"""
    factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                 y::Vector, R::Integer) where T<:Real

Compute the negative log-likelihood for the factor model.

# Arguments
- `θ`: Parameter vector containing:
  * γ parameters (L×J matrix, vectorized): coefficients in measurement equations
  * β parameters (K×1 vector): coefficients in wage equation  
  * α parameters (J+1 vector): factor loadings (J for measurements, 1 for wage)
  * σ parameters (J+1 vector): standard deviations (J for measurements, 1 for wage)
- `X`: Wage equation covariates (N×K)
- `Xfac`: Measurement equation covariates (N×L)
- `Meas`: ASVAB test scores (N×J)
- `y`: Log wages (N×1)
- `R`: Number of quadrature points

# Returns
- Negative log-likelihood value (scalar)

# Model Structure:
## Measurement equations (for each j=1,...,J):
   asvab_j = Xfac*γ_j + α_j*ξ + ε_j,  ε_j ~ N(0, σ_j²)

## Wage equation:
   logwage = X*β + α_{J+1}*ξ + ε,  ε ~ N(0, σ_{J+1}²)

## Latent factor:
   ξ ~ N(0,1)

# Likelihood for person i:
   L_i = ∫ [∏_j φ((M_ij - Xfac_i*γ_j - α_j*ξ)/σ_j) / σ_j] 
          × [φ((y_i - X_i*β - α_{J+1}*ξ)/σ_{J+1}) / σ_{J+1}]
          × φ(ξ) dξ

# Key Steps:
1. Unpack θ into γ, β, α, σ parameters
2. Set up Gauss-Legendre quadrature nodes and weights
3. For each quadrature point:
   a. Compute likelihood contribution from each measurement equation
   b. Compute likelihood contribution from wage equation
   c. Weight by quadrature weight and ξ density
4. Sum log-likelihoods across all observations
5. Return negative for minimization
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                     y::Vector, R::Integer) where T<:Real
    
    # Get dimensions
    K = size(X, 2)      # Number of covariates in wage equation
    L = size(Xfac, 2)   # Number of covariates in measurement equations
    J = size(Meas, 2)   # Number of ASVAB tests
    N = length(y)       # Number of observations
    
    # Unpack parameters from θ vector
    γ = reshape(θ[1:J*L], L, J)  # L×J matrix
    β = θ[J*L+1 : J*L+K]          # K×1 vector
    α = θ[J*L+K+1 : J*L+K+J+1]  # (J+1)×1 vector (factor loadings)
    σ = θ[end - J : end]        # (J+1)×1 vector (standard deviations)
    
    # Get quadrature nodes (ξ) and weights (ω) using lgwt()
    ξ, ω = lgwt(R, -5, 5)
    
    # Initialize likelihood storage
    like = zeros(T, N)
    
    # Loop over quadrature points
    for r in 1:R
        # ASVAB test contribution 
        Mlike = zeros(T, N, J) 
        for j in 1:J
            Mres = Meas[:, j] .- (Xfac * γ[:, j]) .- α[j] * ξ[r]
            sdj = sqrt(σ[j]^2)
            Mlike[:, j] = (1 ./ sdj) .* pdf.(Normal(0, 1), Mres ./ sdj)
            #Mlike[:, j] = pdf.Normal(Mres, sdj)) # maybe similar to above 
        end

        # Wage equation contribution
        yres = y .- (X * β) .- α[end] * ξ[r]
        sdy = sqrt(σ[end]^2)
        Ylike = (1 / sdy) * pdf.(Normal(0, 1), yres ./ sdy)
        #Ylike = pdf.Normal(yres, sdy) # maybe similar to above

        # Construct overall likelihood 
        like += ω[r] .* prod(Mlike, dims = 2) .* Ylike .* pdf.(Normal(0,1), ξ[r])
    end
    
    # Return negative sum of log-likelihoods
    return -sum(log.(like))
end

"""
    run_estimation(df::DataFrame, start_vals::Vector)

Run the full MLE estimation procedure for the factor model.

# Arguments
- `df::DataFrame`: The data
- `start_vals::Vector`: Starting values for optimization

# Returns
- `θ̂`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood at optimum

# Steps:
1. Prepare data matrices
2. Set up TwiceDifferentiable objective (for Hessian-based optimization)
3. Optimize using Newton method with line search
4. Compute standard errors from inverse Hessian
"""
function run_estimation(df::DataFrame, start_vals::Vector)
    # Prepare data matrices
    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    # Optimize
    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, 9), 
                             start_vals; autodiff = :forward)

    result = optimize(
        td,
        start_vals,
        Newton(linesearch = LineSearches.BackTracking()),
        Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true);
        # show_every = 1
    )
    
    # Compute Hessian and standard errors
    H = Optim.hessian!(td, result.minimizer)
    se = sqrt.(diag(inv(H)))
    
    # Return estimates, standard errors, and log-likelihood
    return result.minimizer, se, result.minimum
    
end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main()

Execute the complete analysis workflow for Problem Set 8.

This function runs through all questions sequentially:
1. Load data and run base regression
2. Compute ASVAB correlations
3. Run full regression with all ASVABs
4. Run PCA regression
5. Run Factor Analysis regression
6. Estimate full factor model via MLE
"""
function main()
    # Data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
    
    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)
    
    # Load data
    println("\nLoading data...")
    df = load_data(url)
    println("Data loaded successfully. Dimensions: ", size(df))
     
    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    OLSnoASVAB = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
    println(OLSnoASVAB)
    
    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    # Call compute_asvab_correlations() and display results
    cordf = compute_asvab_correlations(df)
    println(cordf)
    println("Consider: Are these correlations high? What might this imply?")
    
    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    OLSwASVAB = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
                            + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
    println(OLSwASVAB)
    println("Consider: How do results compare to Question 1? Any concerns?")
    println("\nMulticollinearity is a problem; it does not make sense to `hold all 
    else equal` if the focal variable is highly correlated with other variables in 
    the model. They may all be derived from some latent common factor, which is what 
    we should be exogenously varying.")
    
    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    # Call generate_pca!() and display results
    df = generate_pca!(df)
    OLSPCA = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
    println(OLSPCA)
    
   # Question 5
    println("\n" * "="^80)
    println("Question 5: FA Regression")
    println("="^80)
    # Call generate_factor!() and display results
    df = generate_factor!(df)
    OLSFA = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df)
    println(OLSFA)
    
    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    
    # Prepare starting values
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    # - For γ: regress each ASVAB on Xfac
    svals = vcat(
        Xfac\asvabs[:, 1],
        Xfac\asvabs[:, 2],
        Xfac\asvabs[:, 3],
        Xfac\asvabs[:, 4],
        Xfac\asvabs[:, 5],
        Xfac\asvabs[:, 6],
        # - For β: regress wage on X
        X\y,
        # - For α: start with random small values or zeros
        rand(7), 
        # - For σ: start with sample standard deviations or 0.5
        0.5*ones(7)
    )
    
    println("\nEstimating full factor model...")
    θ̂, se, loglike = run_estimation(df, svals)
    println("\nEstimation results:")
    println("Parameter estimates, standard errors, and Z stats:")
    println([θ̂ se θ̂ ./ se])
    println("Log-likelihood at optimum: ", -loglike)
    
    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
    return nothing
end
