# legacy content to facilitate transition to AMP


function _reducePartialManifoldElements(el::Symbol)
  if el == :Euclid
    return TranslationGroup(1)
  elseif el == :Circular
    return Circle()
  end
  error("unknown manifold_symbol $el")
end

"""
    $SIGNATURES

Lots to do here, see RoME.jl #244 and standardized usage with Manifolds.jl.

Notes
- diffop( test, reference )   <===>   ΔX = inverse(test) * reference

DevNotes
- FIXME replace with Manifolds.jl #41, RoME.jl #244
"""
function buildHybridManifoldCallbacks(manif::Tuple)
  # TODO use multiple dispatch instead -- will be done for second version of system
  addopT = []
  diffopT = []
  getManiMu = []
  getManiLam = []

  for mn in manif
    if mn == :Euclid
      push!(addopT, +)
      push!(diffopT, -)
      push!(getManiMu, KDE.getEuclidMu)
      push!(getManiLam, KDE.getEuclidLambda)
    elseif mn == :Circular
      push!(addopT, addtheta)
      push!(diffopT, difftheta)
      push!(getManiMu, getCircMu)
      push!(getManiLam, getCircLambda)
    else
      error("Unrecognized manifold $(mn)")
    end
  end

  return (addopT...,), (diffopT...,), (getManiMu...,), (getManiLam...,)
end

# FIXME TO BE REMOVED
_MtoSymbol(::Euclidean{Tuple{1}}) = :Euclid
_MtoSymbol(::Circle) = :Circular
Base.convert(::Type{<:Tuple}, M::ProductManifold) = _MtoSymbol.(M.manifolds)
Base.convert(::Type{<:Tuple}, M::TranslationGroup) = tuple([:Euclid for i in 1:manifold_dimension(M)]...)

Base.convert(::Type{<:Tuple}, ::Type{<:Manifolds.Euclidean{Tuple{N}, ℝ}} ) where N = tuple([:Euclid for i in 1:N]...)
Base.convert(::Type{<:Tuple}, ::Type{<:Manifolds.Circle{ℝ}})  = error("#FIXME")#(:Circular,)
Base.convert(::Type{<:Tuple}, ::Type{<:Manifolds.RealCircleGroup})  = (:Circular,)
Base.convert(::Type{<:Tuple}, ::Manifolds.Euclidean{Tuple{N}, ℝ} ) where N = tuple([:Euclid for i in 1:N]...)
Base.convert(::Type{<:Tuple}, ::Manifolds.Circle{ℝ})  = error("#FIXME")#(:Circular,)
Base.convert(::Type{<:Tuple}, ::Manifolds.RealCircleGroup)  = (:Circular,)

Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid)}) = (:Euclid,)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid2)}) = (:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid3)}) = (:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid4)}) = (:Euclid,:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(SE2_Manifold)}) = (:Euclid,:Euclid,:Circular)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(SE3_Manifold)}) = (:Euclid,:Euclid,:Euclid,:Circular,:Circular,:Circular)

# Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Manifolds.SpecialOrthogonal(2))}) = (:Circular,)
# Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Manifolds.SpecialOrthogonal(3))}) = (:Circular,:Circular,:Circular)
Base.convert(::Type{<:Tuple}, ::typeof(Manifolds.SpecialOrthogonal(2))) = (:Circular,)
Base.convert(::Type{<:Tuple}, ::typeof(Manifolds.SpecialOrthogonal(3))) = (:Circular,:Circular,:Circular)

Base.convert(::Type{<:Tuple}, ::typeof(Euclid)) = (:Euclid,)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid2)) = (:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid3)) = (:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid4)) = (:Euclid,:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(SE2_Manifold)) = (:Euclid,:Euclid,:Circular)
Base.convert(::Type{<:Tuple}, ::typeof(SE3_Manifold)) = (:Euclid,:Euclid,:Euclid,:Circular,:Circular,:Circular)

"""
    $(SIGNATURES)

Calculate the KDE bandwidths for each dimension independly, as per manifold of each.  Return vector of all dimension bandwidths.
"""
function getKDEManifoldBandwidths(pts::AbstractMatrix{<:Real},
                                  manif::T1 ) where {T1 <: Tuple}
  #
  ndims = size(pts, 1)
  bws = ones(ndims)

  for i in 1:ndims
    if manif[i] == :Euclid
      bws[i] = getBW( kde!(pts[i,:]) )[1,1]
    elseif manif[i] == :Circular
      bws[i] = getBW( kde!_CircularNaiveCV( pts[i,:] ) )[1,1]
    else
      error("Unrecognized manifold $(manif[i])")
    end
  end

  return bws
end



## ================================================================================================================================
# pass through API
## ================================================================================================================================

# not exported yet
# getManifold(x::ManifoldKernelDensity) = x.manifold


import KernelDensityEstimate: Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld


Ndim(x::ManifoldKernelDensity, w...;kw...) = Ndim(x.belief,w...;kw...)
Npts(x::ManifoldKernelDensity, w...;kw...) = Npts(x.belief,w...;kw...)

getWeights(x::ManifoldKernelDensity, w...;kw...) = getWeights(x.belief, w...;kw...)

getKDERange(x::ManifoldKernelDensity, w...;kw...) = getKDERange(x.belief, w...;kw...)
getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...;kw...) = getKDERange((s->s.belief).(x), w...; kw...)
getKDEMax(x::ManifoldKernelDensity, w...;kw...) = getKDEMax(x.belief, w...;kw...)
getKDEMean(x::ManifoldKernelDensity, w...;kw...) = getKDEMean(x.belief, w...;kw...)
getKDEfit(x::ManifoldKernelDensity, w...;kw...) = getKDEfit(x.belief, w...;kw...)

kld(x::ManifoldKernelDensity, w...;kw...) = kld(x.belief, w...;kw...)
minkld(x::ManifoldKernelDensity, w...;kw...) = minkld(x.belief, w...;kw...)

(x::ManifoldKernelDensity)(w...;kw...) = x.belief(w...;kw...)





#