# Interface

import KernelDensityEstimate: getPoints, getBW, Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld
import Random: rand

export getPoints, getBW, Ndims, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export productbelief

struct ManifoldKernelDensity{M <: MB.AbstractManifold{MB.ℝ}, B <: BallTreeDensity, L, P}
  manifold::M
  belief::B
  _partial::L
  """ just an example point for local access to the point data type"""
  _u0::P
end
const MKD{M,B,L} = ManifoldKernelDensity{M, B, L}



function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B,L,P}) where {M,B,L,P}
  printstyled(io, "ManifoldKernelDensity", bold=true, color=:blue )
  printstyled(io, "{$M,", bold=true )
  println(io)
  printstyled(io, "                      $B,", bold=true )
  println(io)
  printstyled(io, "                      $L,", bold=true )
  println(io)
  printstyled(io, "                      $P}(", bold=true )
  println(io)
  println(io, "  dims: ", Ndim(mkd.belief))
  println(io, "  Npts: ", Npts(mkd.belief))
  println(io, "  bws:  ", getBW(mkd.belief)[:,1] .|> x->round(x,digits=4))
  println(io, "  prtl: ", mkd._partial)
  println(io, ")")
  nothing
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)


# override
function marginal(x::ManifoldKernelDensity{M,B}, 
                  dims::AbstractVector{<:Integer}  ) where {M <: AbstractManifold , B}
  #
  ldims::Vector{Int} = collect(dims)
  ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end

function marginal(x::ManifoldKernelDensity{M,B,L}, 
                  dims::AbstractVector{<:Integer}  ) where {M <: AbstractManifold , B, L <: AbstractVector{<:Integer}}
  #
  ldims::Vector{Int} = collect(L[dims])
  ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end
# manis = convert(Tuple, x.manifold)
# partMani = _reducePartialManifoldElements(manis[dims])
# pts = getPoints(x)


"""
    $SIGNATURES

Helper function to convert coordinates to a desired on-manifold point.

Notes
- `u0` is used to identify the data type for a point
- Pass in a different `exp` if needed.
"""
makeManifoldPoint(M::MB.AbstractManifold,
                  coords::AbstractVector{<:Real},
                  u0=zeros(0),
                  ϵ=identity(M,u0),
                  _exp::Function=exp ) = _exp(M, ϵ, hat(M, ϵ, coords))
#

# TODO DEPRECATE
function _matrixCoordsToPoints( M::MB.AbstractManifold, 
                                pts::AbstractMatrix{<:Real}, 
                                u0  )
  #
  @cast ptsArr[j][i] := pts[i,j]
  vecP = Vector{typeof(x._u0)}(undef, length(ptsArr))
  for (j,pt) in enumerate(ptsArr)
    vecP[j] = makeManifoldPoint(M, pt, u0)
  end
  return vecP
end

"""
    $SIGNATURES

Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

Notes
- Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
- Second argument controls whether partial dimensions only should be returned.

"""
function getPoints(x::ManifoldKernelDensity{M,B}, ::Bool=true) where {M <: AbstractManifold, B}
  _matrixCoordsToPoints(x.manifold, getPoints(x.belief), x._u0)
end

function getPoints(x::ManifoldKernelDensity{M,B,L}, aspartial::Bool=true) where {M <: AbstractManifold, B, L <: AbstractVector{Int}}
  pts = getPoints(x.belief)
  pts_ = aspartial ? view(pts, x._partial, :) : pts
  _matrixCoordsToPoints(x.manifold, pts_, x._u0)
end


# TODO check that partials / marginals are sampled correctly
function sample(x::ManifoldKernelDensity{M,B,L,P}, N::Int) where {M,B,L,P}
  # get legacy matrix of coordinates and selected labels
  coords, lbls = sample(x.belief, N)

  # pack samples into vector of point type P
  vecP = Vector{P}(undef, N)
  for j in 1:N
    vecP[j] = makeManifoldPoint(x.manifold, view(coords, :, j), x._u0)
  end

  vecP, lbls
end


function resample(x::ManifoldKernelDensity, N::Int)
  pts = sample(x, N)
  manikde!(x.manifold, pts, x._u0, x._partial)
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



getBW(x::ManifoldKernelDensity, w...;kw...) = getBW(x.belief,w...;kw...)

Ndim(x::ManifoldKernelDensity, w...;kw...) = Ndim(x.belief,w...;kw...)
Npts(x::ManifoldKernelDensity, w...;kw...) = Npts(x.belief,w...;kw...)

getWeights(x::ManifoldKernelDensity, w...;kw...) = getWeights(x.belief, w...;kw...)
Random.rand(x::ManifoldKernelDensity, d::Integer=1) = rand(x.belief, d)

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


# NOTE, this product does not handle combinations of different partial beliefs properly yet
function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct(PP, PP[1].manifold)
end

function *(P1::MKD{M,B}, P2::MKD{M,B}, P_...) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct([P1;P2;P_...], P1.manifold)
end



# # take the full pGM in, but only update the coordinate dimensions that are actually affected by new information.
# function _partialProducts!( pGM::AbstractVector{P}, 
#                             partials::Dict{Any, <:AbstractVector{<:ManifoldKernelDensity}},
#                             manifold::MB.AbstractManifold; 
#                             inclFull::Bool=true  ) where P <: AbstractVector
#   #
#   # manis = convert(Tuple, manifold)
#   keepold = inclFull ? deepcopy(pGM) : typeof(pGM)()

#   # TODO remove requirement for P <: AbstractVector
#   allPartDimsMask = 0 .== zeros(Int, length(pGM[1]))
#   # FIXME, remove temporary Tuple manifolds method 
#   for (dimnum,pp) in partials
#     # mark dimensions getting partial information
#     for d in dimnum
#       allPartDimsMask[d] = true
#     end
#     # change to vector
#     dimv = [dimnum...]
#     # include previous calcs (if full density products were done before partials)
#     partialMani = _buildManifoldPartial(manifold, dimv)
#     # take product of this partial's subset of dimensions
#     partial_GM = AMP.manifoldProduct(pp, partialMani, Niter=1) |> getPoints
#     # partial_GM = AMP.manifoldProduct(pp, (manis[dimv]...,), Niter=1) |> getPoints
    
#     for i in 1:length(pGM)
#       pGM[i][dimv] = partial_GM[i]
#     end
#   end
  
#   # multiply together previous full dim and new product of various partials
#   if inclFull
#     partialPts = [pGM[i][dimv] for i in 1:length(pGM)]
#     push!( pp, AMP.manikde!(partialPts, partialMani) )
#   end
  

#   nothing
# end


"""
    $SIGNATURES

Take product of `dens` (including optional partials beliefs) as proposals to be multiplied together.

Notes
-----
- Return points of full dimension, even if only partial dimensions in proposals.
  - 'Other' dimensions left unchanged from incoming `denspts`
- `d` dimensional product approximation
- `partials` are treated per each unique Tuple subgrouping, i.e. (1,2), (2,), ...
- Incorporate ApproxManifoldProducts to process variables in individual batches.

DevNotes
- TODO Consolidate with [`AMP.manifoldProduct`](@ref), especially concerning partials. 
"""
function productbelief( denspts::AbstractVector{P},
                        manifold::MB.AbstractManifold,
                        dens::Vector{<:ManifoldKernelDensity},
                        # partials::Dict{Any, <:AbstractVector{<:ManifoldKernelDensity}},
                        N::Int;
                        asPartial::Bool=false,
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P
  #
  # TODO only works of P <: Vector
  Ndens = length(dens)
  # Npartials = length(partials)
  Ndims = maximum(Ndim.(dens))
  with_logger(logger) do
    @info "[$(Ndens)x,d$(Ndims),N$(N)],"
  end
  
  # # resize for #1013
  # if size(denspts,2) < N
  #   pGM = zeros(size(denspts,1),N)
  #   pGM[:,1:size(denspts,2)] .= denspts
  # else
  #   pGM = deepcopy(denspts)
  # end

  mkd = AMP.manifoldProduct(dens, manifold, Niter=1, oldPoints=denspts)
  pGM = getPoints(mkd, asPartial)

  # # TODO VALIDATE inclFull is the right order
  # (pGM, inclFull) = if 0 < Ndens
  #   getPoints(AMP.manifoldProduct(dens, manifold, Niter=1)), true # false
  # elseif Ndens == 0 && 0 < Npartials
  #   deepcopy(denspts), false # true
  # else
  #   error("Unknown density product Ndens=$(Ndens), Npartials=$(Npartials)")
  # end

  # # take the product between partial dimensions
  # _partialProducts!(pGM, partials, manifold, inclFull=inclFull)

  return pGM
end




#