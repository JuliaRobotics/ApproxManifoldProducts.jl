module ApproxManifoldProducts

import Base: *, isapprox, convert, show, eltype, length

using Logging
using StaticArrays
using LinearAlgebra
import LinearAlgebra: rotate!, det

using TensorCast
using DocStringExtensions
using Distributions
using Random
import Random: rand

using Statistics
import Statistics: mean, std, cov, var, entropy

import Rotations as _Rot
using CoordinateTransformations

using RecursiveArrayTools: ArrayPartition
export ArrayPartition

import ManifoldsBase
import ManifoldsBase: AbstractManifold, distance
using Manifolds

using NLsolve
import Optim

import TransformUtils as TUs
import TransformUtils: skew

using Reexport
@reexport using KernelDensityEstimate
import KernelDensityEstimate: getPoints, getBW, evalAvgLogL, entropy, evaluate


# FIXME ON FIRE OBSOLETE REMOVE
using Requires

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

include("services/ManifoldsOverloads.jl")
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
