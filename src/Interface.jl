# Interface

mutable struct ManifoldKernelDensity{M <: MB.Manifold{MB.ℝ}, B <: BallTreeDensity}
  manifold::Type{M}
  belief::B
end
const MKD{M,B} = ManifoldKernelDensity{M, B}

# ManifoldKernelDensity(m::M,b::B) where {M <: MB.Manifold{MB.ℝ}, B} = ManifoldKernelDensity{M,B}(m,b)


function ManifoldKernelDensity(m::Type{<:MB.Manifold}, pts::AbstractArray{<:Real})
  tup = convert(Tuple, m)
  bel = manikde!(pts, m)
  ManifoldKernelDensity(m, bel)
end


@deprecate ManifoldBelief(w...;kw...) ManifoldKernelDensity(w...;kw...)
function ManifoldBelief(::Type{<:M}, mkd::ManifoldKernelDensity{M,T}) where {M <: MB.Manifold{MB.ℝ}, T} 
  @warn "ManifoldBelief is deprecated, use ManifoldKernelDensity instead"
  return mkd
end

function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B}) where {M, B}
  printstyled(io, "ManifoldKernelDensity{$M,$B}(\n", bold=true)
  show(io, mkd.belief)
  println(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)

function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.Manifold{MB.ℝ},B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end

function *(P1::MKD{M,B}, P2::MKD{M,B}) where {M<:MB.Manifold{MB.ℝ},B}
  # @info "taking manifold product of $(length(PP)) terms, $M, $B"
  # @error "No known product definition"
  manis = convert(Tuple, M)
  manifoldProduct([P1.belief;P2.belief], manis)
end


## ================================================================================================================================
# Serialization
## ================================================================================================================================

# abstract type JSONManifoldKernelDensity end

# export JSONManifoldKernelDensity

function Base.convert(::Type{<:AbstractString}, 
                      mkd::ManifoldKernelDensity)
  #
  dict = Dict{Symbol, String}()
  dict[:_type] = "ManifoldKernelDensity"
  dict[:belief] = KDE.string( mkd.belief )
  dict[:manifold] = string(mkd.manifold)

  JSON2.write(dict)
end

function Base.convert(::Type{<:ManifoldKernelDensity}, str::AbstractString)
  dict = JSON2.read(str)
  # FIXME, make module specific
  manis = getfield(Main, Symbol(dict[:manifold]))
  belief_ = convert(BallTreeDensity, dict[:belief])
  ManifoldKernelDensity(manis, belief_)
end


## ================================================================================================================================
# pass through API
## ================================================================================================================================

# not exported yet
getManifold(x::ManifoldKernelDensity) = x.manifold


import KernelDensityEstimate: getPoints, getBW, Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld
import Random: rand

export getPoints, getBW, Ndims, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld


getPoints(x::ManifoldKernelDensity, w...;kw...) = getPoints(x.belief,w...;kw...)
getBW(x::ManifoldKernelDensity, w...;kw...) = getBW(x.belief,w...;kw...)

Ndims(x::ManifoldKernelDensity, w...;kw...) = Ndims(x.belief,w...;kw...)
Npts(x::ManifoldKernelDensity, w...;kw...) = Npts(x.belief,w...;kw...)

getWeights(x::ManifoldKernelDensity, w...;kw...) = getWeights(x.belief, w...;kw...)
marginal(x::ManifoldKernelDensity, w...;kw...) = marginal(x.belief, w...;kw...)
sample(x::ManifoldKernelDensity, w...;kw...) = sample(x.belief, w...;kw...)
Random.rand(x::ManifoldKernelDensity, d::Integer=1) = rand(x.belief, d)
resample(x::ManifoldKernelDensity, w...;kw...) = resample(x.belief, w...;kw...)

getKDERange(x::ManifoldKernelDensity, w...;kw...) = getKDERange(x.belief, w...;kw...)
getKDEMax(x::ManifoldKernelDensity, w...;kw...) = getKDEMax(x.belief, w...;kw...)
getKDEMean(x::ManifoldKernelDensity, w...;kw...) = getKDEMean(x.belief, w...;kw...)
getKDEfit(x::ManifoldKernelDensity, w...;kw...) = getKDEfit(x.belief, w...;kw...)

kld(x::ManifoldKernelDensity, w...;kw...) = kld(x.belief, w...;kw...)
minkld(x::ManifoldKernelDensity, w...;kw...) = minkld(x.belief, w...;kw...)

(x::ManifoldKernelDensity)(w...;kw...) = x.belief(w...;kw...)