

# A few supported manifolds

export 
  Euclid,
  EuclideanManifold,
  Euclid2,
  SE2_Manifold,
  SE3_Manifold


# abstract type ManifoldDefs end
#
# FIXME standardize with Manifolds.jl
struct Euclid <: MB.Manifold{MB.ℝ} end
# TODO, consolidate
struct EuclideanManifold <: MB.Manifold{MB.ℝ} end
struct Euclid3 <: MB.Manifold{MB.ℝ} end
struct Euclid4 <: MB.Manifold{MB.ℝ} end

struct SE2_Manifold <: MB.Manifold{MB.ℝ} end
struct SE3_Manifold <: MB.Manifold{MB.ℝ} end

struct SE2E2_Manifold <: MB.Manifold{MB.ℝ} end
