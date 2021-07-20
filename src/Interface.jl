# Interface

import ManifoldsBase: identity

export makeCoordsFromPoint, makePointFromCoords, getNumberCoords
export identity

# Deprecate in favor of TranslationGroup instead
ManifoldsBase.identity(::Euclidean{Tuple{N}}, val::AbstractVector{T}) where {N, T <: Number} = zeros(T, N)
ManifoldsBase.identity(::Circle, val::AbstractVector{T}) where {T <: Real} = zeros(T, 1)

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
                    ϵ=identity(M,u0),
                    retraction_method::AbstractRetractionMethod=ExponentialRetraction()  ) = retract(M, ϵ, hat(M, ϵ, coords), retraction_method)
#

function makeCoordsFromPoint( M::MB.AbstractManifold,
                              pt::P ) where P
  #
  # only works for manifold which have an identity (eg groups)
  ϵ = identity(M, pt)
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
  ϵ = identity(M, pts[1])
  for (j,pt) in enumerate(pts)
    mat[:,j] = vee(M, ϵ, log(M, ϵ, pt))
  end

  return mat
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