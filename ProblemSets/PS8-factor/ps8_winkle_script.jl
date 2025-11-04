using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

# Set working directory
cd(@__DIR__)

include("ps8_winkle_source.jl")

main()
