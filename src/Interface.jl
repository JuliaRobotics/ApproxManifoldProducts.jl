# Interface

ManifoldsBase.identity(::Euclidean{Tuple{N}}, val::AbstractVector{T}) where{N, T} = zeros(T, N)

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
                    retraction_method::AbstractRetractionMethod=ExponentialRetraction() ) = retract(M, ϵ, hat(M, ϵ, coords), retraction_method)
#


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


coords(::Type{<:typeof(SpecialEuclidean(2))}, p::ProductRepr) = [p.parts[1][1], p.parts[1][2], atan(p.parts[2][2,1],p.parts[2][1,1])]

function uncoords(::Type{<:typeof(SpecialEuclidean(2))}, p::AbstractVector{<:Real}, static::Bool=true)
  α = p[3] 
  ArrConst = static ? SA : eltype(α)
  return ProductRepr((ArrConst[p[1], p[2]]), ArrConst[cos(α) -sin(α); sin(α) cos(α)])
end
# function uncoords(::Type{<:typeof(SpecialEuclidean(2))}, p::AbstractVector{<:Real})
#   α = p[3]
#   return ProductRepr(([p[1], p[2]]), [cos(α) -sin(α); sin(α) cos(α)])
# end

function coords(::Type{<:typeof(SpecialEuclidean(3))}, p::ProductRepr)
  wELo = TU.convert(Euler, SO3(p.parts[2]))
  [p.parts[1][1:3]; wELo.R; wELo.P; wELo.Y]
end

function uncoords(::Type{<:typeof(SpecialEuclidean(3))}, p::AbstractVector{<:Real})
  # α = p[3]
  wRo = TU.convert(SO3, Euler(p[4:6]...))
  return ProductRepr(([p[1], p[2], p[3]]), wRo.R)
end




# TODO, hack, use the proper Manifolds.jl intended vectoration methods instead
_makeVectorManifold(::MB.AbstractManifold, arr::AbstractArray{<:Real}) = arr
_makeVectorManifold(::MB.AbstractManifold, val::Real) = [val;]
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(2))} = coords(M, prr)
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(3))} = coords(M, prr)



#