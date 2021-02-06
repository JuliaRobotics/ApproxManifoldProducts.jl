# Interface

mutable struct ManifoldKernelDensity{M <: MB.Manifold{MB.ℝ}, B}
  manifold::Type{M}
  belief::B
end
ManifoldKernelDensity(m::M,b::B) where {M <: MB.Manifold{MB.ℝ}, B} = ManifoldKernelDensity{M,B}

const MKD{M,B} = ManifoldKernelDensity{M, B}

@deprecate ManifoldBelief(w...;kw...) ManifoldKernelDensity(w...;kw...)



function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.Manifold{MB.ℝ},B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end


## ================================================================================================================================
# pass through API
## ================================================================================================================================

# not exported yet
getManifold(x::ManifoldKernelDensity) = x.manifold


import KernelDensityEstimate: getPoints, getBW, Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample

export getPoints, getBW, Ndims, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample


getPoints(x::ManifoldKernelDensity, w...;kw...) = getPoints(x.belief,w...;kw...)
getBW(x::ManifoldKernelDensity, w...;kw...) = getBW(x.belief,w...;kw...)

Ndims(x::ManifoldKernelDensity, w...;kw...) = Ndims(x.belief,w...;kw...)
Npts(x::ManifoldKernelDensity, w...;kw...) = Npts(x.belief,w...;kw...)

getWeights(x::ManifoldKernelDensity, w...;kw...) = getWeights(x.belief, w...;kw...)
marginal(x::ManifoldKernelDensity, w...;kw...) = marginal(x.belief, w...;kw...)
sample(x::ManifoldKernelDensity, w...;kw...) = sample(x.belief, w...;kw...)
rand(x::ManifoldKernelDensity, w...;kw...) = rand(x.belief, w...;kw...)
resample(x::ManifoldKernelDensity, w...;kw...) = resample(x.belief, w...;kw...)

getKDERange(x::ManifoldKernelDensity, w...;kw...) = getKDERange(x.belief, w...;kw...)
getKDEMax(x::ManifoldKernelDensity, w...;kw...) = getKDEMax(x.belief, w...;kw...)
getKDEMean(x::ManifoldKernelDensity, w...;kw...) = getKDEMean(x.belief, w...;kw...)
getKDEfit(x::ManifoldKernelDensity, w...;kw...) = getKDEfit(x.belief, w...;kw...)


(x::ManifoldKernelDensity)(w...;kw...) = x.belief(w...;kw...)