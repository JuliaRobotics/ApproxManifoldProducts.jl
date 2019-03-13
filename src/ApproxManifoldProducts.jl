module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
@reexport using TransformUtils

using DocStringExtensions

using NLsolve
using Optim
using CoordinateTransformations
using Requires

import Base: *
# import KernelDensityEstimate: kde!

const KDE = KernelDensityEstimate
const TUs = TransformUtils
const CTs = CoordinateTransformations

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

export
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
  ensurePeriodicDomains!,
  solveresid2DLinear!,
  solveresid2DLinear,
  ManifoldBelief,
  MB,
  *,

  # APi and util functions
  buildHybridManifoldCallbacks,
  getKDEManifoldBandwidths,
  manifoldProduct,
  manikde!,

  # Supported manifolds
  Manifold,
  EuclideanManifold,
  Euclid2,
  Circular


include("Interface.jl")
include("CommonUtils.jl")
include("Euclidean.jl")
include("CircularUtils.jl")
include("Circular.jl")

include("Legacy.jl")
include("API.jl")

function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
