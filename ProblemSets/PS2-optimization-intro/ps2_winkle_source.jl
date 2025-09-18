function ps2()
    #-------------------------------------------------------------------------------
    # question 1
    #-------------------------------------------------------------------------------

    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, BFGS())
    println("argmin (minimizer) is ",Optim.minimizer(result)[1])
    println("min is ",Optim.minimum(result)) 

    result_better = optim = optimize(minusf, [-7.0], BFGS())
    println(result_better)

    #-------------------------------------------------------------------------------
    # question 2
    #-------------------------------------------------------------------------------

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    # estmate optimal parameters using OLS 
    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.     Options(g_tol=1e-6, iterations=100_000, show_trace=true))

    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y

    # standard errors
    N = size(X,1)
    K = size(X,2)
    mse = sum((y .- X*bols).^2)/(N - K)
    vcov = mse*inv(X'*X)
    se_bols = sqrt.(diag(vcov))
    println("standard errors: ", se_bols)

    println("OLS closed` form solution: ", bols)
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)


    #---------------------------------------------------------------------------
    # question 3
    #---------------------------------------------------------------------------

    function logit(alpha, X, d)
        Xa = X * alpha
        p = 1 ./(1 .+ exp.(-Xa))
        loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
        return -loglike
    end

    alpha_hat_logit = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

    println(alpha_hat_logit.minimizer)

    # alternative implementation by dr ransom 
    function logit(alpha, X, y)
        P = 1 ./(1 .+ exp.(-X * alpha))
        loglike = -sum((y.==1) .* log.(P) .+ (y.==0) .* log.(1 .- P))
        return loglike
    end

    alpha_hat_optim = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

    println(alpha_hat_optim.minimizer)

    #---------------------------------------------------------------------------
    # question 4
    #---------------------------------------------------------------------------

    df.white = df.race .== 1

    println("Logit Optim solution: ", Optim.minimizer(alpha_hat_logit))

    logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

    isapprox(Optim.minimizer(alpha_hat_logit), coef(logit_glm); atol=1e-6)

    # compare with dr ransom's implementation
    isapprox(alpha_hat_optim.minimizer, coef(logit_glm); atol=1e-6)

    #---------------------------------------------------------------------------
    # question 5
    #---------------------------------------------------------------------------

    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j in 1:J
            bigY[:,j] = y .== j
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

        num = zeros(N,J)
        denom = zeros(N)
        for j in 1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            denom += num[:,j]
        end

        P = num ./ repeat(denom, 1, J)

        loglike = -sum(bigY .* log.(P))

        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    akpha_rand = rand(6*size(X,2))

    alpha_true = [
        # constant, age, white, collegegrad,
        0.1910213, -0.0335262,  0.5963968,  0.4165052,   # pro/technical
        -0.1698368, -0.0359784,  1.30684,   -0.430997,    # managers/admin
        0.6894727, -0.0104578,  0.5231634, -1.492475,    # sales
        -2.26748,   -0.0053001,  1.391402, -0.9849661,    # clerical/unskilled
        -1.398468,  -0.0142969, -0.0176531, -1.495123,    # craftsmen
        0.2454891, -0.0067267, -0.5382892, -3.78975      # operatives
    ]

    alpha_start = alpha_true .* rand(size(alpha_true))
    alpha_hat_mlogit = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println("Multinomial logit estimates: ", alpha_hat_mlogit.minimizer)

    #---------------------------------------------------------------------------
    # clean print
    #---------------------------------------------------------------------------

    println("OLS closed` form solution: ", bols)
    println("Logit Optim solution: ", Optim.minimizer(alpha_hat_logit))
    println("Logit GLM solution: ", coef(logit_glm))
    println("MLogit Optim solution: ", alpha_hat_mlogit.minimizer)

    return nothing
end


