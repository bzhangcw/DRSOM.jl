###############
# project: RSOM
# created Date: Tu Mar 2022
# author: <<author>
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
using .LP


using LIBSVMFileIO

bool_plot = true
bool_opt = false

n = 500
m = 1000
μ = 5e-2
X = [rand(Float64, n) * 2 .- 1 for _ in 1:m]
Xm = hcat(X)'
y = rand(Float64, m) * 2 .- 1
# y = max.(y, 0)
# loss
x0 = ones(n)/10
function loss(w)
    loss_single(x, y0) = exp((w' * x - y0) / μ)
    _pure = loss_single.(X, y) |> sum
    return μ * log(_pure)
end
function grad(w)
    a = (Xm * w - y) / μ
    ax = exp.(a)
    π0 = ax / (ax |> sum)
    ∇ = Xm' * π0
    return ∇
end
function hess(w)
    a = (Xm * w - y) / μ
    ax = exp.(a)
    π0 = ax / (ax |> sum)
    return 1 / μ * (Xm' * Diagonal(π0) * Xm - Xm' * π0 * π0' * Xm)
end


if bool_opt
    # compare with GD and LBFGS, Trust region newton,
    options = Optim.Options(
        g_tol=1e-6,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=10,
        time_limit=500
    )
    r_lbfgs = Optim.optimize(
        loss, grad, x0,
        LBFGS(; alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.BackTracking()), options;
        inplace=false
    )
    r_newton = Optim.optimize(
        loss, grad, hess, x0,
        Newton(; alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.HagerZhang()), options;
        inplace=false
    )
    r = HSODM()(;
        x0=copy(x0), f=loss, g=grad, H=hess,
        maxiter=10000, tol=1e-6, freq=1,
        direction=:warm, linesearch=:hagerzhang
    )
    r.name = "Adaptive HSODM"
    rh = PFH()(;
        x0=copy(x0), f=loss, g=grad, H=hess,
        maxiter=10000, tol=1e-6, freq=1,
        step=:hsodm, μ₀=5e-1,
        bool_trace=true,
        maxtime=10000,
        direction=:warm
    )
end

if bool_plot

    results = [
        optim_to_result(r_lbfgs, "LBFGS"),
        optim_to_result(r_newton, "Newton's method"),
        r,
        rh
    ]
    method_names = getname.(results)
    # for metric in (:ϵ, :fx)
    metric = :ϵ
        method_objval_ragged = rstack([
                getresultfield.(results, metric)...
            ]; fill=NaN
        )


        @printf("plotting results\n")

        pgfplotsx()
        title = L"Soft Maximum"
        fig = plot(
            1:(method_objval_ragged|>size|>first),
            method_objval_ragged,
            label=permutedims(method_names),
            xscale=:log2,
            yscale=:log10,
            xlabel="Iteration",
            ylabel=metric == :ϵ ? L"\|\nabla f\| = \epsilon" : L"f(x)",
            title=title,
            size=(1280, 720),
            yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
            xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
            dpi=1000,
        )

        savefig(fig, "/tmp/$metric-softmaximum.pdf")

    end
end