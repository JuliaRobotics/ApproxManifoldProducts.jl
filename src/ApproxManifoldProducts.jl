module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate

using CoordinateTransformations
using TransformUtils

using Requires

using NLsolve
using Optim

import Base: *

const KDE = KernelDensityEstimate
const TUs = TransformUtils
const CTs = CoordinateTransformations

export
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
  solveresid2DLinear!,
  solveresid2DLinear,
  ManifoldBelief,
  MB,
  *,

  # Supported manifolds
  Manifold,
  EuclideanManifold,
  Euclid2,
  Circular


include("Interface.jl")
include("CommonUtils.jl")
include("Euclidean.jl")
include("Circular.jl")


function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
