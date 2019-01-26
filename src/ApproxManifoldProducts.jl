module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate

using Requires

using NLsolve
using Optim

import Base: *

export
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
  solveresid2DLinear!,
  solveresid2DLinear,
  ManifoldBelief,
  MB,
  *,

  # Supported manifolds
  Manifolds,
  EuclideanManifold

abstract type Manifolds end

struct EuclideanManifold <: Manifolds
end


mutable struct ManifoldBelief{M <: Manifolds, B}
  manifold::Type{M}
  belief::B
end
ManifoldBelief(m::M,b::B) where {M <: Manifolds, B} = ManifoldBelief{M,B}

const MB{M,B} = ManifoldBelief{M, B}

function *(PP::Vector{MB{M,B}}) where {M<:Manifolds,B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end

function *(PP::Vector{MB{EuclideanManifold,BallTreeDensity}})
  bds = Vector{BallTreeDensity}(undef, length(PP))
  for p in PP
    bds[i] = p.belief
  end
  *(bds)
end


get2DLambda(Lambdas::Vector{Float64})::Float64 = sum(Lambdas)

function resid2DLinear(μ, mus, Lambdas; diffop::Function=-)  # '-' exploits EuclideanManifold commutativity a-b = b-a
  dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
  # @show round.(dμ, digits=4)
  ret = sum( Lambdas.*dμ )
  return ret
end

function solveresid2DLinear!(res, x, mus, Lambdas; diffop::Function=-)::Nothing
  res[1] = resid2DLinear(x, mus, Lambdas, diffop=diffop)
  nothing
end

# import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function=-)::Float64
  solveresid2DLinear!(res, x, mus, Lambdas, diffop=diffop)
  return res[1]
end


function get2DMu(mus, Lambdas; diffop::Function=-, periodicmanifold::Function=(x)->x, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5) )::Float64
  # TODO: can be solved as the null space basis, but requires proper scaling
  gg = (res, x) -> solveresid2DLinear!(res, x, mus, Lambdas, diffop=diffop)
  initr = initrange[2]-initrange[1]
  x0 = [initr*rand()+initrange[1]]
  r = NLsolve.nlsolve(gg, x0)
  xs = sign(r.zero[1] - x0[1])*1e-3 .+ r.zero
  r = NLsolve.nlsolve(gg, xs)
  return periodicmanifold(r.zero[1])
end


function get2DMuMin(mus, Lambdas; diffop::Function=-, periodicmanifold::Function=(x)->x, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5), method=Optim.Newton(), Λμ::Bool=false )::Float64
  # TODO: can be solved as the null space basis, but requires proper scaling
  res = zeros(1)
  # @show round.(mus, digits=3)
  gg = (x) -> (solveresid2DLinear(res, x, mus, Lambdas, diffop=diffop))^2
  initr = initrange[2]-initrange[1]
  # TODO -- do we need periodicmanifold here?
  x0 = [initr*rand()+initrange[1]]
  r = Optim.optimize(gg, x0, method)
  # @show r.minimizer[1]
  return periodicmanifold(r.minimizer[1])
end


function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
