using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("ps3_winkle_source.jl")

allwrap()

#-------------------------------------------------------------------------------
# interpret the estimated coefﬁcient γ-hat. 
#-------------------------------------------------------------------------------

# Estimated gamma is the coefficient on the log of income (the final n element).γ-hat = -0.09419, which represents the change in latent utility with a 1-unit change in the relative expected log wage in occupation j (relative to other). This result is surprising, given that we would expect people would prefer earning more money than earning less.