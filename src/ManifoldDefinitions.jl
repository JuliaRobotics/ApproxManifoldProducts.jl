

# A few supported manifolds

export 
  # general manifolds
  Euclid,
  Euclid2,
  Euclid3,
  Euclid4,
  SE2_Manifold,
  SE3_Manifold,
  
  # special quirks that still need to be fixed
  Circle1
  # Circular,

#


# this is just wrong and needs to be fixed
const Circle1 = Circle{ℝ}

const Euclid =  Euclidean{Tuple{1}, ℝ} # Euclidean(1)
const EuclideanManifold = Euclid

const Euclid2 = Euclidean{Tuple{2}, ℝ} # Euclidean(2)
const Euclid3 = Euclidean{Tuple{3}, ℝ} # Euclidean(3)
const Euclid4 = Euclidean{Tuple{4}, ℝ} # Euclidean(4)

# TODO if not easy simplification exists, then just deprecate this
const SE2_Manifold = typeof(SpecialEuclidean(2))
const SE3_Manifold = typeof(SpecialEuclidean(3))


Base.convert(::Type{B}, mkd::ManifoldKernelDensity{M,B}) where {M,B<:BallTreeDensity} = mkd.belief



# # abstract type ManifoldDefs end
# #
# # FIXME standardize with Manifolds.jl
# struct Euclid <: MB.Manifold{MB.ℝ} end
# struct Euclid3 <: MB.Manifold{MB.ℝ} end
# struct Euclid4 <: MB.Manifold{MB.ℝ} end
# struct Sphere1 <: MB.Manifold{MB.ℝ} end
# struct SE2_Manifold <: MB.Manifold{MB.ℝ} end
# struct SE3_Manifold <: MB.Manifold{MB.ℝ} end

# # Deprecate below

# export EuclideanManifold

# const EuclideanManifold = Euclid
# # @deprecate EuclideanManifold() Euclid() 
# # struct EuclideanManifold <: MB.Manifold{MB.ℝ} end



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

function uncoords(::Type{<:typeof(SpecialEuclidean(2))}, p::AbstractVector{<:Real})
  α = p[3]
  return ProductRepr(([p[1], p[2]]), [cos(α) -sin(α); sin(α) cos(α)])
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









#