# Interface

import Base: replace
import ManifoldsBase: identity

export makeCoordsFromPoint, makePointFromCoords, getNumberCoords
export identity
export replace

# Deprecate in favor of TranslationGroup instead, also type piracy
Manifolds.identity_element(::Euclidean{Tuple{N}}, val::AbstractVector{T}) where {N, T <: Number} = zeros(T, N)
Manifolds.identity_element(::Circle, val::AbstractVector{T}) where {T <: Real} = zeros(T, 1)

"""
    $SIGNATURES

Helper function to convert coordinates to a desired on-manifold point.

Notes
- `u0` is used to identify the data type for a point
- Pass in a different `exp` if needed.
"""
makePointFromCoords(M::MB.AbstractManifold,
                    coords::AbstractVector{<:Real},
                    u0=zeros(manifold_dimension(M)),
                    ϵ=identity_element(M,u0),
                    retraction_method::AbstractRetractionMethod=ExponentialRetraction()  ) = retract(M, ϵ, hat(M, ϵ, coords), retraction_method)
#

function makeCoordsFromPoint( M::MB.AbstractManifold,
                              pt::P ) where P
  #
  # only works for manifold which have an identity (eg groups)
  ϵ = identity_element(M, pt)
  vee(M, ϵ, log(M, ϵ, pt))
end

# Sphere(2) has 3 coords, even though the manifolds only has 2 dimensions (degrees of freedom)
getNumberCoords(M::MB.AbstractManifold, p) = length(makeCoordsFromPoint(M,p))

# TODO DEPRECATE
# related _pointsToMatrixCoords
function _matrixCoordsToPoints( M::MB.AbstractManifold, 
                                pts::AbstractMatrix{<:Real}, 
                                u0  )
  #
  # ptsArr = Vector{Vector{Float64}}(undef, size(pts, 2))
  # @cast ptsArr[j][i] = pts[i,j]
  vecP = Vector{typeof(u0)}(undef, size(pts, 2))
  for j in 1:size(pts,2)
    pt = pts[:,j]
    vecP[j] = makePointFromCoords(M, pt, u0)
  end
  return vecP
end


function _pointsToMatrixCoords(M::MB.AbstractManifold, pts::AbstractVector{P}) where P
  mat = zeros(manifold_dimension(M), length(pts))
  ϵ = identity_element(M, pts[1])
  for (j,pt) in enumerate(pts)
    mat[:,j] = vee(M, ϵ, log(M, ϵ, pt))
  end

  return mat
end



# default replace non-partial/non-marginal values
# Trivial case where no information from destination is kept, only from src.
function Base.replace(::ManifoldKernelDensity{M,<:BallTreeDensity,Nothing}, 
                      src::ManifoldKernelDensity{M,<:BallTreeDensity,Nothing} 
                      ) where {M<:AbstractManifold}
  #
  src
end

# replace dest non-partial with incoming partial values
function Base.replace( dest::ManifoldKernelDensity{M,<:BallTreeDensity,Nothing}, 
                        src::ManifoldKernelDensity{M,<:BallTreeDensity,<:AbstractVector} 
                      ) where {M<:AbstractManifold}
  #
  pl = src._partial
  oldPts = getPoints(dest.belief)
  # get source partial points only 
  newPts = getPoints(src.belief)
  @assert size(newPts,2) == size(oldPts,2) "this replace currently requires the number of points to be the same, dest=$(size(oldPts,2)), src=$(size(newPts,2))"
  for i in 1:size(oldPts, 2)
    oldPts[pl,i] .= newPts[pl,i]
  end
  # and new bandwidth
  oldBw = getBW(dest.belief)[:,1]
  oldBw[pl] .= getBW(src.belief)[pl,1]

  # finaly update the belief with a new container
  newBel = kde!(oldPts, oldBw)

  # also set the metadata values
  ipc = deepcopy(dest.infoPerCoord)
  ipc[pl] .= src.infoPerCoord[pl]
  
  # and _u0 point is a bit more tricky
  c0 = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, dest._u0))
  c_ = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, src._u0))
  c0[pl] .= c_[pl]
  u0 = exp(dest.manifold, dest._u0, hat(dest.manifold, dest._u0, c0))

  # return the update destimation ManifoldKernelDensity object
  ManifoldKernelDensity(dest.manifold, newBel, nothing, u0, infoPerCoord=ipc)
end


# replace partial/marginal with different incoming partial values
function Base.replace(dest::ManifoldKernelDensity{M,<:BallTreeDensity,<:AbstractVector}, 
                      src::ManifoldKernelDensity{M,<:BallTreeDensity,<:AbstractVector} 
                      ) where {M<:AbstractManifold}
  #
  pl = src._partial
  oldPts = getPoints(dest.belief)
  # get source partial points only 
  newPts = getPoints(src.belief)
  @assert size(newPts,2) == size(oldPts,2) "this replace currently requires the number of points to be the same, dest=$(size(oldPts,2)), src=$(size(newPts,2))"
  for i in 1:size(oldPts, 2)
    oldPts[pl,i] .= newPts[pl,i]
  end
  # and new bandwidth
  oldBw = getBW(dest.belief)[:,1]
  oldBw[pl] .= getBW(src.belief)[pl,1]

  # finaly update the belief with a new container
  newBel = kde!(oldPts, oldBw)

  # also set the metadata values
  ipc = deepcopy(dest.infoPerCoord)
  ipc[pl] .= src.infoPerCoord[pl]

  # and _u0 point is a bit more tricky
  c0 = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, dest._u0))
  c_ = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, src._u0))
  c0[pl] .= c_[pl]
  u0 = exp(dest.manifold, dest._u0, hat(dest.manifold, dest._u0, c0))

  # and update the partial information
  pl_ = union(dest._partial, pl)

  # return the update destimation ManifoldKernelDensity object
  if length(pl_) == manifold_dimension(dest.manifold)
    # no longer a partial/marginal
    return ManifoldKernelDensity(dest.manifold, newBel, nothing, u0, infoPerCoord=ipc)
  else
    # still a partial
    return ManifoldKernelDensity(dest.manifold, newBel, pl_, u0, infoPerCoord=ipc)
  end
end


##============================================================================================================
## New Manifolds.jl aware API -- TODO find the right file placement
##============================================================================================================



# TODO, hack, use the proper Manifolds.jl intended vectoration methods instead
_makeVectorManifold(::MB.AbstractManifold, arr::AbstractArray{<:Real}) = arr
_makeVectorManifold(::MB.AbstractManifold, val::Real) = [val;]
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(2))} = coords(M, prr)
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(3))} = coords(M, prr)



#