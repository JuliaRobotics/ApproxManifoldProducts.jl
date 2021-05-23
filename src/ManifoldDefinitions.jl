

# A few supported manifolds

export 
  # general manifolds
  Euclidean,
  Circle,
  ℝ,
  Euclid,
  Euclid2,
  Euclid3,
  Euclid4,
  SE2_Manifold,
  SE3_Manifold,
  
  # special quirks that still need to be fixed
  Circle1
  # Circular,

export
  coords,
  uncoords,
  getPointsManifold,
  calcMean

#


# this is just wrong and needs to be fixed
const Circle1 = Circle()

const Euclid =  Euclidean(1)
const EuclideanManifold = Euclid

const Euclid2 = Euclidean(2)
const Euclid3 = Euclidean(3)
const Euclid4 = Euclidean(4)

# TODO if not easy simplification exists, then just deprecate this
const SE2_Manifold = SpecialEuclidean(2)
const SE3_Manifold = SpecialEuclidean(3)


Base.convert(::Type{B}, mkd::ManifoldKernelDensity{M,B}) where {M,B<:BallTreeDensity} = mkd.belief



##============================================================================================================
## New Manifolds.jl aware API -- TODO find the right file placement
##============================================================================================================


function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: Euclidean}
  data_ = getPoints(mkd.belief)
  TensorCast.@cast data[i][j] := data_[j,i]
  return data
end

function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: Circle}
  data_ = getPoints(mkd.belief)
  return data_[:]
end

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

function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: SpecialEuclidean}
  data_ = getPoints(mkd.belief)
  [uncoords(M, view(data_, :, i)) for i in 1:size(data_,2)]
end


# TODO, hack, use the proper Manifolds.jl intended vectoration methods instead
_makeVectorManifold(::MB.AbstractManifold, arr::AbstractArray{<:Real}) = arr
_makeVectorManifold(::MB.AbstractManifold, val::Real) = [val;]
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(2))} = coords(M, prr)
_makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(3))} = coords(M, prr)


function calcMean(mkd::ManifoldKernelDensity{M}) where {M <: ManifoldsBase.AbstractManifold}
  data = getPointsManifold(mkd)
  mprepr = mean(mkd.manifold, data)
  
  #
  _makeVectorManifold(mkd.manifold, mprepr)
end








#