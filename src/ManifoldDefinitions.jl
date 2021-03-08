

# A few supported manifolds

export 
  Euclid,
  Euclid2,
  Euclid3,
  Euclid4,
  Sphere1,
  SE2_Manifold,
  SE3_Manifold,
  SE2E2_Manifold


# abstract type ManifoldDefs end
#
# FIXME standardize with Manifolds.jl
struct Euclid <: MB.Manifold{MB.ℝ} end
struct Euclid3 <: MB.Manifold{MB.ℝ} end
struct Euclid4 <: MB.Manifold{MB.ℝ} end
struct Sphere1 <: MB.Manifold{MB.ℝ} end
struct SE2_Manifold <: MB.Manifold{MB.ℝ} end
struct SE3_Manifold <: MB.Manifold{MB.ℝ} end
struct SE2E2_Manifold <: MB.Manifold{MB.ℝ} end


# Deprecate below

export EuclideanManifold

const EuclideanManifold = Euclid
@deprecate EuclideanManifold() Euclid() 
# struct EuclideanManifold <: MB.Manifold{MB.ℝ} end
