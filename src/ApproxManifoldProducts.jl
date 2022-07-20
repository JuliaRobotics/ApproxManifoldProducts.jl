module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
@reexport using TransformUtils  # likely to be deprecated
using Random

import ManifoldsBase
import ManifoldsBase: AbstractManifold
using RecursiveArrayTools: ArrayPartition
export ArrayPartition

using Manifolds

using DocStringExtensions

using NLsolve
import Optim
using CoordinateTransformations
using Requires
using LinearAlgebra
using TensorCast
using StaticArrays
using Logging
using Statistics
using NearestNeighbors

import Random: rand

import Base: *, isapprox, convert
import LinearAlgebra: rotate!
import Statistics: mean, std, cov, var
import KernelDensityEstimate: getPoints, getBW
import TransformUtils: rotate!

const NNR = NearestNeighbors
const MB = ManifoldsBase
const CTs = CoordinateTransformations
const AMP = ApproxManifoldProducts
const KDE = KernelDensityEstimate
const TUs = TransformUtils

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

# the exported API
include("ExportAPI.jl")

# internal features not exported
include("_BiMaps.jl")

# AMP types and some legacy support 
include("entities/ManifoldKernelDensity.jl")
include("entities/ManifoldDefinitions.jl")
include("Legacy.jl")
include("services/ManifoldPartials.jl")
include("Interface.jl")

# regular features
include("CommonUtils.jl")
include("services/ManifoldKernelDensity.jl")
include("services/Euclidean.jl")
include("services/CircularUtils.jl")
include("services/Circular.jl")
include("KernelHilbertEmbeddings.jl")

include("TrackingLabels.jl")

# include("Serialization.jl") # moved downstream to IIF to use InferenceVariable serialized types instead
include("API.jl")

include("Deprecated.jl")

function __init__()
  @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
    @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
  end
end


end
