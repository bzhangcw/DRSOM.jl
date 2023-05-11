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
using SparseArrays
using .LP
using LoopVectorization
using LIBSVMFileIO

bool_plot = false
bool_opt = false
bool_q_preprocessed = true

name = "a4a"
# name = "w4a"
# name = "covtype"
# Load data
X, y = libsvmread("test/instances/libsvm/$name.libsvm"; dense=false)
Xv = hcat(X...)'
y = convert(Vector{Float64}, y)

@info begin
    a = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    "using BLAS threads $a"
end
@info "data reading finished"

# precompute Q Matrix
Qf(x, y) = y^2 * x * x'
Qc(x, y) = y^2 * x
bool_q_preprocessed && (Q = Qf.(X, y))
bool_q_preprocessed && (P = Qc.(X, y))
Pv = hcat(P...)'
# loss
λ = 1e-5
n = X[1] |> length
Random.seed!(1)
N = y |> length



function sloss(w)
    loss_single(x, y) = log(1 + exp(-y * w' * x))
    _pure = loss_single.(X, y) |> sum
    return _pure / N + 0.5 * λ * w'w
end
function loss(w)
    _pure = vmapreduce(
        (x, y) -> log(1 + exp(-y * w' * x)),
        +,
        X,
        y
    )
    return _pure / N + 0.5 * λ * w'w
end
function g(w)
    function _g(x, y, w)
        ff = exp(-y * w' * x)
        return -ff / (1 + ff) * y * x
    end
    _pure = vmapreduce(
        (x, y) -> _g(x, y, w),
        +,
        X,
        y
    )
    return _pure / N + λ * w
end
function H(w)
    # function _H(x, y, q, w)
    #     ff = exp(-y * w' * x)
    #     return ff / (1 + ff)^2 * q
    # end
    # _pure = vmapreduce(
    #     (x, y, q) -> _H(x, y, q, w),
    #     +,
    #     X,
    #     y,
    #     Q
    # )
    # return _pure / N + λ * I
    z = exp.(y .* (Xv * w))
    fq = z ./ (1 .+ z) .^ 2
    return ((fq .* Pv)' * Xv ./ N) + λ * I
end
function hvp(w, v, Hv; eps=1e-8)
    function _hvp(x, y, q, w, v)
        wx = w' * x
        ff = exp(-y * wx)
        return ff / (1 + ff)^2 * q * x' * v
    end
    _pure = vmapreduce(
        (x, y, q) -> _hvp(x, y, q, w, v),
        +,
        X,
        y,
        P
    )
    # copyto!(Hv, 1 / eps .* g(w + eps .* v) - 1 / eps .* g(w))
    copyto!(Hv, _pure ./ N .+ λ .* v)
end

function hvp1(w, v, Hv)
    z = exp.(y .* (Xv * w))
    fq = z ./ (1 .+ z) .^ 2
    copyto!(Hv, (fq .* Pv)' * (Xv * v) ./ N .+ λ .* v)
end
@info "data preparation finished"
x0 = 0 * randn(Float64, n)
ε = 1e-8 * max(g(x0) |> norm, 1)


if bool_opt

    # options for Optim.jl package
    options = Optim.Options(
        g_tol=1e-6,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )

    r_newton = Optim.optimize(
        loss, g, H, x0,
        Newton(; alphaguess=LineSearches.InitialStatic()), options;
        inplace=false
    )

    r_lbfgs = Optim.optimize(
        loss, g, x0,
        LBFGS(; alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.BackTracking()), options;
        inplace=false
    )

    # r = HSODM()(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvp,
    #     maxiter=10000, tol=1e-6, freq=1,
    #     maxtime=10000,
    #     direction=:warm, linesearch=:hagerzhang,
    #     adaptive=:none
    # )

    rn = PFH()(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=10000, tol=ε, freq=1,
        step=:newton, μ₀=5e-2,
        maxtime=10000,
        direction=:warm
    )
    # rh = PFH()(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvp,# H=H,
    #     maxiter=10000, tol=ε, freq=1,
    #     step=:hsodm, μ₀=5e-1,
    #     maxtime=10000,
    #     direction=:warm
    # )
    rh = PFH()(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=10000, tol=ε, freq=1,
        step=:hsodm, μ₀=5e-1,
        bool_trace=false,
        maxtime=10000,
        direction=:warm
    )

end


if bool_plot

    results = [
        # optim_to_result(res1, "GD+Wolfe"),
        # optim_to_result(res2, "LBFGS+Wolfe"),
        optim_to_result(r_newton, "Newton's method"),
        r,
    ]
    method_names = getname.(results)
    for metric in (:ϵ, :fx)
        method_objval_ragged = rstack([
                getresultfield.(results, metric)...
            ]; fill=NaN
        )


        @printf("plotting results\n")

        pgfplotsx()
        title = L"\min _{w \in {R}^{d}} \frac{1}{2} \sum_{i=1}^{n}\left(\frac{1}{1+e^{-w^{\top} x_{i}}}-y_{i}\right)^{2}+\frac{\alpha}{2}\|w\|^{2}"
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

        savefig(fig, "/tmp/$metric-logistic-$name.pdf")

    end
end
