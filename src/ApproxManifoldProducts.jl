module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
@reexport using TransformUtils

using DocStringExtensions

using NLsolve
using Optim
using CoordinateTransformations
using Requires
using SLEEFPirates
using LinearAlgebra

import Base: *, isapprox
# import KernelDensityEstimate: kde!
import LinearAlgebra: rotate!

const AMP = ApproxManifoldProducts
const KDE = KernelDensityEstimate
const TUs = TransformUtils
const CTs = CoordinateTransformations

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

export
  AMP,
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
  isapprox,

  # APi and util functions
  buildHybridManifoldCallbacks,
  getKDEManifoldBandwidths,
  manifoldProduct,
  manikde!,

  # general manifolds
  Manifold,
  Circular

# internal features not exported
include("_BiMaps.jl")

# regular features
include("Interface.jl")
include("CommonUtils.jl")
include("Euclidean.jl")
include("CircularUtils.jl")
include("Circular.jl")
include("KernelHilbertEmbeddings.jl")

include("Legacy.jl")
include("API.jl")

function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
