# Interface

export productbelief

mutable struct ManifoldKernelDensity{M <: MB.Manifold{MB.ℝ}, B <: BallTreeDensity}
  manifold::M
  belief::B
end
const MKD{M,B} = ManifoldKernelDensity{M, B}

# ManifoldKernelDensity(m::M,b::B) where {M <: MB.Manifold{MB.ℝ}, B} = ManifoldKernelDensity{M,B}(m,b)


function ManifoldKernelDensity(m::MB.Manifold, pts::AbstractArray{<:Real})
  tup = convert(Tuple, m)
  bel = manikde!(pts, m)
  ManifoldKernelDensity(m, bel)
end


@deprecate ManifoldBelief(w...;kw...) ManifoldKernelDensity(w...;kw...)
function ManifoldBelief(::M, mkd::ManifoldKernelDensity{M,T}) where {M <: MB.Manifold{MB.ℝ}, T} 
  @warn "ManifoldBelief is deprecated, use ManifoldKernelDensity instead"
  return mkd
end

function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B}) where {M, B}
  printstyled(io, "ManifoldKernelDensity{$M,$B}(\n", bold=true )
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
  # make module specific
  # good references: 
  #  https://discourse.julialang.org/t/converting-string-to-datatype-with-meta-parse/33024/2
  #  https://discourse.julialang.org/t/is-there-a-way-to-import-modules-with-a-string/15723/6
  manisplit = split(dict[:manifold], '.')
  manimod, manitp = if 1 < length(manisplit)
    modex = Symbol(manisplit[1])
    @eval($modex), dict[:manifold][(length(manisplit[1])+2):end]
  else
    Main, dict[:manifold]
  end
  manip = Meta.parse(manitp)
  manis = Core.eval(manimod, manip) # could not get @eval to work with $
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



## ================================================================================================================================
# Legacy Interface for product of full and partial dimensions
## ================================================================================================================================




function _partialProducts!(pGM, partials, manis; useExisting::Bool=false)
  for (dimnum,pp) in partials
    dimv = [dimnum...]
    # include previous calcs (if full density products were done before partials)
    !useExisting ? nothing : push!(pp, AMP.manikde!(pGM[dimv,:], (manis[dimv]...,) ))
    # take product of this partial's subset of dimensions
    pGM[dimv,:] = AMP.manifoldProduct(pp, (manis[dimv]...,), Niter=1) |> getPoints
  end
  nothing
end


"""
    $SIGNATURES

Take product of `dens` accompanied by optional `partials` proposal belief densities.

Notes
-----
- `d` dimensional product approximation
- `partials` are treated per each unique Tuple subgrouping, i.e. (1,2), (2,), ...
- Incorporate ApproxManifoldProducts to process variables in individual batches.
"""
function productbelief( denspts::AbstractArray,
                        manifold::ManifoldsBase.Manifold,
                        dens::Vector{<:BallTreeDensity},
                        partials::Dict{Any, <:AbstractVector{<:BallTreeDensity}},
                        N::Int;
                        dbg::Bool=false,
                        logger=ConsoleLogger()  )
  #
  manis = manifold |> getManifolds
  lennonp, lenpart = length(dens), length(partials)
  Ndims = size(denspts,1)
  with_logger(logger) do
    @info "[$(lennonp)x$(lenpart)p,d$(Ndims),N$(N)],"
  end
  
  # # resize for #1013
  # if size(denspts,2) < N
  #   pGM = zeros(size(denspts,1),N)
  #   pGM[:,1:size(denspts,2)] .= denspts
  # else
  #   pGM = deepcopy(denspts)
  # end
  
  # pGM = Array{Float64,2}(undef, 0,0)
  # new, slightly condensed partialProduct operation
  (pGM, uE) = if 0 < lennonp
    getPoints(AMP.manifoldProduct(dens, manis, Niter=1)), true
  elseif lennonp == 0 && 0 < lenpart
    deepcopy(denspts), false
  else
    error("Unknown density product lennonp=$(lennonp), lenpart=$(lenpart)")
  end
  # take the product between partial dimensions
  _partialProducts!(pGM, partials, manis, useExisting=uE)

  return pGM
end




#