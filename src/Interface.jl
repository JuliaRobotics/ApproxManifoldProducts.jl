# Interface

export productbelief

mutable struct ManifoldKernelDensity{M <: MB.AbstractManifold{MB.ℝ}, B <: BallTreeDensity}
  manifold::M
  belief::B
end
const MKD{M,B} = ManifoldKernelDensity{M, B}

# ManifoldKernelDensity(m::M,b::B) where {M <: MB.AbstractManifold{MB.ℝ}, B} = ManifoldKernelDensity{M,B}(m,b)


function ManifoldKernelDensity( M::MB.AbstractManifold,
                                ptsArr::AbstractVector{P},
                                bw::AbstractVector{<:Real}  ) where P
  #
  # FIXME obsolete
  arr = Matrix{Float64}(undef, length(ptsArr[1]), length(ptsArr))
  @cast arr[i,j] = ptsArr[j][i]
  manis = convert(Tuple, M)
  addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
  bel = KernelDensityEstimate.kde!(arr, bw, addopT, diffopT)
  return ManifoldKernelDensity(M, bel)
end

function ManifoldKernelDensity( M::MB.AbstractManifold, 
                                ptsArr::AbstractVector{P} ) where P
  #
  # FIXME obsolete
  arr = Matrix{Float64}(undef, length(ptsArr[1]), length(ptsArr))
  @cast arr[i,j] = ptsArr[j][i]
  manis = convert(Tuple, M)
  bw = getKDEManifoldBandwidths(arr, manis )
  addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
  bel = KernelDensityEstimate.kde!(arr, bw, addopT, diffopT)
  return ManifoldKernelDensity(M, bel)
end


function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B}) where {M, B}
  printstyled(io, "ManifoldKernelDensity{$M,$B}(\n", bold=true )
  show(io, mkd.belief)
  println(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)

function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.AbstractManifold{MB.ℝ},B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end

function *(P1::MKD{M,B}, P2::MKD{M,B}) where {M<:MB.AbstractManifold{MB.ℝ},B}
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
# getManifold(x::ManifoldKernelDensity) = x.manifold


import KernelDensityEstimate: getPoints, getBW, Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld
import Random: rand

export getPoints, getBW, Ndims, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld

# with DFG v0.15 change points to Vector{P}
function getPoints(x::ManifoldKernelDensity, w...;kw...) 
  pts = getPoints(x.belief,w...;kw...)
  @cast ptsArr[j][i] := pts[i,j]
  return ptsArr
end


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

# legacy

"""
    $SIGNATURES
Interface function to return the `variableType` manifolds of an InferenceVariable, extend this function for all Types<:InferenceVariable.
"""
function getManifolds end

getManifolds(::Type{<:T}) where {T <: ManifoldsBase.AbstractManifold} = convert(Tuple, T)
getManifolds(::T) where {T <: ManifoldsBase.AbstractManifold} = getManifolds(T)


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
function productbelief( denspts::AbstractVector{P},
                        manifold::ManifoldsBase.AbstractManifold,
                        dens::Vector{<:BallTreeDensity},
                        partials::Dict{Any, <:AbstractVector{<:BallTreeDensity}},
                        N::Int;
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P <: AbstractVector{<:Real}
  #
  manis = manifold |> getManifolds
  # TODO only works of P <: Vector
  @show typeof(dens), typeof(partials)
  @show lennonp = Ndim(dens[1])
  @show lenpart = length(partials)
  Ndims = size(denspts[1])
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

  @cast pGM_Arr[j][i] := pGM[i,j]

  return pGM_Arr
end




#