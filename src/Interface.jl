# Interface

import Base: replace
export makeCoordsFromPoint, makePointFromCoords, getNumberCoords
export identity_element
export setPointPartial!, setPointsMani!
export replace


"""
    $SIGNATURES

Helper function to convert coordinates to a desired on-manifold point.

DevNotes
- FIXME need much better consolidation or even removal of this function entirely.
  - This function makes too strong an assumption of groups

Notes
- `u0` is used to identify the data type for a point
- Pass in a different `exp` if needed.
"""
makePointFromCoords(M::MB.AbstractManifold,  # Manifolds.AbstractGroupManifold
                    coords::AbstractVector{<:Real},
                    u0=zeros(manifold_dimension(M)),
                    ϵ=identity_element(M,u0),
                    retraction_method::AbstractRetractionMethod=ExponentialRetraction()  ) = retract(M, ϵ, hat(M, ϵ, coords), retraction_method)
#

# should perhaps just be dispatched for <:AbstractGroupManifold
# only works for AbstractGroupManifold (have an identity)
function makeCoordsFromPoint( M::MB.AbstractManifold,
                              pt::P,
                              ϵ = identity_element(M, pt) ) where P
  #
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

# asPartial=true indicates that src coords are smaller than dest coords, and false implying src has dummy values in placeholder dimensions
function setPointPartial!(Mdest::AbstractManifold, 
                          dest, 
                          Msrc::AbstractManifold, 
                          src, 
                          partial::AbstractVector{<:Integer},
                          asPartial::Bool=true )
  #
  # trivial case of empty factor
  if length(partial) == 0
    return dest
  end

  dest_ = AMP.makeCoordsFromPoint(Mdest,dest)
  # e0 = identity_element(Mdest, dest)
  # dest_ = vee(Mdest, e0, log(Mdest, e0, dest))
  
  # Note on partial cases.
  #  Mdest is always full dimensional as the destination of some new values.
  #  Msrc is partial dimension manifold.
  #  src is assumed to be values which only represent the partial values 

  # FIXME, does this line need to cater for both partial and tangent or point cases?
  src_ = AMP.makeCoordsFromPoint(Msrc,src)
  # e0s = identity_element(Msrc, src)
  # src_ = vee(Msrc, e0s, log(Msrc, e0s, src))

  # do the copy in coords 
  dest_[partial] .= asPartial ? src_ : view(src_, partial)
  
  # update points base in original
  dest__ = makePointFromCoords(Mdest, dest_, dest)
  # dest__ = exp(Mdest, e0, hat(Mdest, e0, dest_))
  setPointsMani!(dest, dest__)
  
  #
  return dest 
end

function setPointPartial!(
  Mdest::AbstractManifold, 
  dest::AbstractArray{T}, 
  Msrc::AbstractManifold, 
  src::AbstractArray{U},
  partial::AbstractVector{<:Integer},
  destIdx,
  srcIdx=destIdx,
  asPartial::Bool=true
) where {T<:AbstractArray,U<:AbstractArray}

  if isbitstype(T)
    #TODO needs cleanup, this is copied from setPointPartial! above with index changes
    if length(partial) == 0
      return dest[destIdx]
    end
    dest_coords = collect(AMP.makeCoordsFromPoint(Mdest, dest[destIdx]))
    src_coords = AMP.makeCoordsFromPoint(Msrc, src[srcIdx])
    dest_coords[partial] .= asPartial ? src_coords : view(src_coords, partial)
    return dest[destIdx] = makePointFromCoords(Mdest, dest_coords, dest[destIdx])
    
  else
    return setPointPartial!(Mdest, dest[destIdx], Msrc, src[srcIdx], partial, asPartial)
  end

end

#TODO workaround for supporting bitstypes, need rewrite, can consider `PowerManifoldNestedReplacing` or similar
function setPointsMani!(dest::AbstractArray{T}, src::AbstractArray{U}, destIdx, srcIdx=destIdx) where {T<:AbstractArray,U<:AbstractArray}
  if isbitstype(T)
    dest[destIdx] = src[srcIdx]
  else
    setPointsMani!(dest[destIdx],src[srcIdx])
  end
end


#TODO  ArrayPartition should work for now as it's an AbstractVector, but it won't remain mutable
setPointsMani!(dest::AbstractVector, src::AbstractVector) = (dest .= src)
setPointsMani!(dest::AbstractMatrix, src::AbstractMatrix) = (dest .= src)
function setPointsMani!(dest::AbstractVector, src::AbstractMatrix)
  @assert size(src,2) == 1 "Workaround setPointsMani! currently only allows size(::Matrix, 2) == 1"
  setPointsMani!(dest, src[:])
end
function setPointsMani!(dest::AbstractMatrix, src::AbstractVector)
  @assert size(dest,2) == 1 "Workaround setPointsMani! currently only allows size(::Matrix, 2) == 1"
  setPointsMani!(view(dest,:,1), src)
end

function setPointsMani!(dest::AbstractVector, src::AbstractVector{<:AbstractVector})
  @assert length(src) == 1 "Workaround setPointsMani! currently only allows Vector{Vector{P}}(...) |> length == 1"
  setPointsMani!(dest, src[1])
end

function setPointsMani!(dest::ProductRepr, src::ProductRepr)
  for (k,prt) in enumerate(dest.parts)
    setPointsMani!(prt, src.parts[k])
  end
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
  c0 = collect(vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, dest._u0)))
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





#