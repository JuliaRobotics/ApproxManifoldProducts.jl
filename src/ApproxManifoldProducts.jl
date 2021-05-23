module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
@reexport using TransformUtils
using Random

import ManifoldsBase
import ManifoldsBase: AbstractManifold
const MB = ManifoldsBase
using Manifolds

using DocStringExtensions

using NLsolve
import Optim
using CoordinateTransformations
using Requires
using SLEEFPirates
using LinearAlgebra
using JSON2
using TensorCast
using StaticArrays
using Logging

import Base: *, isapprox, convert
# import KernelDensityEstimate: kde!
import LinearAlgebra: rotate!


const AMP = ApproxManifoldProducts
const KDE = KernelDensityEstimate
const TUs = TransformUtils
const CTs = CoordinateTransformations

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

export  
  # new local features
  AMP,
  MKD,
  AbstractManifold,
  ManifoldKernelDensity,
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
  ensurePeriodicDomains!,
  solveresid2DLinear!,
  solveresid2DLinear,
  *,
  isapprox,

  # APi and util functions
  buildHybridManifoldCallbacks,
  getKDEManifoldBandwidths,
  manifoldProduct,
  manikde!


# internal features not exported
include("_BiMaps.jl")

# regular features
include("Interface.jl")
include("CommonUtils.jl")
include("ManifoldDefinitions.jl")
include("Euclidean.jl")
include("CircularUtils.jl")
include("Circular.jl")
include("KernelHilbertEmbeddings.jl")

include("TrackingLabels.jl")

include("Legacy.jl")
include("API.jl")

function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
