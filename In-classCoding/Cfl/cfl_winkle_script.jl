using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, ForwardDiff, FreqTables, Distributions

cd(@__DIR__)

include("cfl_winkle_source.jl")

main_cfls()