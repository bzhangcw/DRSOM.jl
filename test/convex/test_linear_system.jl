#############################################
# project: DRSOM
# created Date: Tu Mar 2022
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test DRSOM on nonconvex logistic regression for 0-1 classification on LIBSVM
# @reference:
# 1. Zhu, X., Han, J., Jiang, B.: An Adaptive High Order Method for Finding Third-Order Critical Points of Nonconvex Optimization, http://arxiv.org/abs/2008.04191, (2020)
###############


include("../lp.jl")
include("../tools.jl")

using ArgParse
using DRSOM
using Distributions
using LineSearches
using Optim
using ProximalOperators
using ProximalAlgorithms
using Random
using Plots
using Printf
using LazyStack
using KrylovKit
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using Optim
using SparseArrays
using DRSOM
using .LP
using LoopVectorization
using LIBSVMFileIO
using DataFrames
using CSV
Base.@kwdef mutable struct KrylovInfo
    normres::Float64
    numops::Float64
end
table = []
# name = "a4a"
# name = "a9a"
# name = "w4a"
# name = "covtype"
# name = "news20"
# name = "rcv1"
# names = ["a4a"] #, "a9a", "w4a", "covtype", "rcv1", "news20"]
names = ["a4a", "a9a", "w4a", "covtype", "rcv1"]
@warn "news20 is very big..., consider run on a server"
# use the data matrix of libsvm
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))
Random.seed!(1)

for name in names
    @info "run $name"
    X, y = libsvmread("test/instances/libsvm/$name.libsvm"; dense=false)
    Xv = hcat(X...)'
    Rc = 1 ./ f1(Xv)[:]
    Xv = (Rc |> Diagonal) * Xv
    if name in ["covtype"]
        y = convert(Vector{Float64}, (y .- 1.5) * 2)
    else
    end

    γ = 1e-3
    n = Xv[1, :] |> length
    Random.seed!(1)
    N = y |> length

    Q = Xv' * Xv
    function gfs(w)
        return (Q * w - Xv' * y) / N + γ * w
    end
    r = Dict()
    # 
    r["GHM-Lanczos"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-CG"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-GMRES"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-rGMRES"] = KrylovInfo(normres=0.0, numops=0)

    samples = 5
    for idx in 1:samples
        w₀ = rand(Float64, n)
        g = gfs(w₀)
        δ = -0.1
        ϵᵧ = 1e-5
        function hvp(w)
            gn = gfs(w₀ + ϵᵧ .* w)
            gf = g
            return (gn - gf) / ϵᵧ
        end
        Fw(w) = [hvp(w[1:end-1]) + g .* w[end]; w[1:end-1]' * g + δ * w[end]]
        Fc = DRSOM.Counting(Fw)
        Hc = DRSOM.Counting(hvp)
        @info "data reading finished"

        max_iteration = 200
        ε = 1e-6

        rl = KrylovKit.eigsolve(
            Fc, [w₀; 1], 1, :SR, Lanczos(tol=ε / 10, maxiter=max_iteration, verbosity=3, eager=true);
        )
        λ₁ = rl[1]
        ξ₁ = rl[2][1]

        r["GHM-Lanczos"].normres += (Fc(ξ₁) - λ₁ .* ξ₁) |> norm
        r["GHM-Lanczos"].numops += rl[end].numops

        rl = KrylovKit.linsolve(
            Hc, -g, w₀, CG(; tol=ε, maxiter=max_iteration, verbosity=3);
        )
        r["Newton-CG"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-CG"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=ε, maxiter=1, krylovdim=max_iteration, verbosity=3);
        )
        r["Newton-GMRES"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-GMRES"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=ε, maxiter=4, krylovdim=div(max_iteration, 4), verbosity=3);
        )
        r["Newton-rGMRES"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-rGMRES"].numops += rl[end].numops
    end
    for (k, v) in r
        push!(table, [name, n, k, v.numops / samples, v.normres / samples])
    end
end

tmat = hcat(table...)
df = DataFrame(
    name=tmat[1, :],
    dim=tmat[2, :],
    method=tmat[3, :],
    k=tmat[4, :],
    ϵ=tmat[5, :]
)

CSV.write("/tmp/linsys.csv", df)

"""
import pandas as pd
df = pd.read_csv("/tmp/linsys.csv")
print(df.set_index(["name", "method"]).to_latex(multirow=True, longtable=True))
"""