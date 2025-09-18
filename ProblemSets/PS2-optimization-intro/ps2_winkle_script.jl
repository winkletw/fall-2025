using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# read in the function
include("ps2_winkle_source.jl")

# execute the function
ps2()

