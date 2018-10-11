module ApproxManifoldProducts

using NLsolve

import Base: *

export
  get2DLambda,
  get2DMu,
  resid2DLinear,
  solveresid2DLinear,
  ManifoldBelief,
  *

abstract type Manifolds end

struct EuclideanManifold <: Manifolds
end

struct SO2Manifold <: Manifolds
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

function *(PP::Vector{MB{EuclideanManifold,B}}) where B
  @info "taking manifold product of $(length(PP)) terms"
  @warn "EuclideanManifold: work in progress"
end
function *(PP::Vector{MB{SO2Manifold,B}}) where B
  @info "taking manifold product of $(length(PP)) terms"
  @warn "SO2Manifold: work in progress"
end


get2DLambda(Lambdas) = sum(Lambdas)

function resid2DLinear(μ, mus, Lambdas; diffop::Function=-)
  dμ = broadcast(diffop, mus, μ)  # mus .- μ
  @show round.(dμ, digits=4)
  ret = sum( Lambdas.*dμ )
  return ret
end

# import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function=-)
  res[1] = resid2DLinear(x, mus, Lambdas, diffop=diffop)
end

function get2DMu(mus, Lambdas; diffop::Function=-)
  # TODO: can be solved as the null space basis, but requires proper scaling
  gg = (res, x) -> solveresid2DLinear(res, x, mus, Lambdas, diffop=diffop)
  r = NLsolve.nlsolve(gg, [1e-5*randn()])
  return r.zero
end

end
