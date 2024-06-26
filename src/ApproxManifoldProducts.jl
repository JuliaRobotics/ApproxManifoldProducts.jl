module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
using Random

import TransformUtils as TUs
import Rotations as _Rot

import ManifoldsBase
import ManifoldsBase: AbstractManifold, distance
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
using Distributions

import Random: rand

import Base: *, isapprox, convert, show, eltype, length
import LinearAlgebra: rotate!, det
import Statistics: mean, std, cov, var, entropy
import KernelDensityEstimate: getPoints, getBW, evalAvgLogL, entropy, evaluate

const MB = ManifoldsBase
const CTs = CoordinateTransformations
const AMP = ApproxManifoldProducts
const KDE = KernelDensityEstimate

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

# the exported API
include("ExportAPI.jl")

# internal features not exported
include("_BiMaps.jl")

include("entities/KernelEval.jl")
include("entities/ManellicTree.jl") # experimental
include("entities/ManifoldKernelDensity.jl")

include("CommonUtils.jl")
include("services/ManellicTree.jl")

# AMP types and some legacy support 
include("entities/ManifoldDefinitions.jl")
include("Legacy.jl")
include("services/ManifoldPartials.jl")
include("Interface.jl")

# regular features
include("services/KernelEval.jl")
include("services/ManifoldKernelDensity.jl")
include("services/Euclidean.jl")
include("services/CircularUtils.jl")
include("services/Circular.jl")
include("KernelHilbertEmbeddings.jl")

include("TrackingLabels.jl")

# include("Serialization.jl") # moved downstream to IIF to use InferenceVariable serialized types instead
include("API.jl")

include("Deprecated.jl")

# weak dependencies
include("../ext/WeakdepsPrototypes.jl")


end
