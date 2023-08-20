module ApproxManifoldProducts

using Reexport
@reexport using KernelDensityEstimate
using Random

import TransformUtils as TUs
import Rotations as _Rot

import ManifoldsBase
import ManifoldsBase: AbstractManifold
# using RecursiveArrayTools: ArrayPartition

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

import NearestNeighbors as NNR
import Distances as DST


import Base: *, isapprox, convert
import LinearAlgebra: rotate!
import Statistics: mean, std, cov, var
import Random: rand

import NearestNeighbors: TreeData, NNTree
import Manifolds: ArrayPartition

import KernelDensityEstimate: getPoints, getBW

const MB = ManifoldsBase
const CTs = CoordinateTransformations
const AMP = ApproxManifoldProducts
const KDE = KernelDensityEstimate

# TODO temporary for initial version of on-manifold products
KDE.setForceEvalDirect!(true)

export ArrayPartition

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

# Experimental ManifoldBallTreeBalanced
include("services/TreeDataBalanced.jl")
include("services/ManifoldTreeOps.jl")
include("services/ManifoldHyperSpheres.jl")
include("services/ManifoldBallTree.jl")

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

# weak dependencies
include("../ext/WeakdepsPrototypes.jl")


end
