module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate

using NLsolve
using Optim

import Base: *

export
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
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


get2DLambda(Lambdas) = sum(Lambdas)

function resid2DLinear(μ, mus, Lambdas; diffop::Function=-)  # '-' exploits EuclideanManifold commutativity a-b = b-a
  dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
  # @show round.(dμ, digits=4)
  ret = sum( Lambdas.*dμ )
  return ret
end

# import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function=-)
  res[1] = resid2DLinear(x, mus, Lambdas, diffop=diffop)
end

function get2DMu(mus, Lambdas; diffop::Function=-, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5) )
  # TODO: can be solved as the null space basis, but requires proper scaling
  gg = (res, x) -> solveresid2DLinear(res, x, mus, Lambdas, diffop=diffop)
  r = NLsolve.nlsolve(gg, [initr*rand()+initrange[1]])
  return r.zero
end


function get2DMuMin(mus, Lambdas; diffop::Function=-, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5), method=Optim.Newton() )
  # TODO: can be solved as the null space basis, but requires proper scaling
  res = zeros(1)
  gg = (x) -> (solveresid2DLinear(res, x, mus, Lambdas, diffop=diffop))^2
  initr = initrange[2]-initrange[1]
  x0 = [initr*rand()+initrange[1]]
  r = Optim.optimize(gg, x0, method)
  return r.minimizer
end

end
