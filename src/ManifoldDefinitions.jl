

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
  Sphere1,
  Circular,
  SE2E2_Manifold

#


# this is a hack
struct SE2E2_Manifold <: MB.Manifold{MB.ℝ} end
# const SE2E2_Manifold = 

# this is just wrong and needs to be fixed
const Sphere1 = Circle{ℝ}

const Euclid =  Euclidean{Tuple{1}, ℝ} # Euclidean(1)
const EuclideanManifold = Euclid

const Euclid2 = Euclidean{Tuple{2}, ℝ} # Euclidean(2)
const Euclid3 = Euclidean{Tuple{3}, ℝ} # Euclidean(3)
const Euclid4 = Euclidean{Tuple{4}, ℝ} # Euclidean(4)

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
