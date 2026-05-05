module ApproxManifoldProducts

import Base: *, isapprox, convert, show, eltype, length
import Base: isassigned

using Logging
using StaticArrays
import SparseArrays
using LinearAlgebra
import LinearAlgebra: rotate!, det

using TensorCast
using DocStringExtensions
using Distributions
using Random
import Random: rand

using Statistics
import Statistics: mean, std, cov, var # , entropy? 24Q3, # JL v1.11-rc1, Statistics v1.11.1.

import Rotations as _Rot
using CoordinateTransformations

using RecursiveArrayTools: ArrayPartition
export ArrayPartition

using PDMats: PDMat

using ManifoldsBase
using ManifoldsBase: AbstractManifold, distance, TypeParameter, submanifold_component
import Manifolds
using LieGroups
using LieGroups: TranslationGroup

import Optim


const MB = ManifoldsBase
const CTs = CoordinateTransformations


include("../ext/HomotopyRepr.jl")

# the exported API
include("ExportAPI.jl")

# internal features not exported
include("_BiMaps.jl")

include("entities/KernelEval.jl")
include("entities/HomotopyDensity.jl")

include("services/ManifoldsOverloads.jl")
include("services/EigenSortUtils.jl")
include("services/HomotopyTreeUtils.jl")
include("services/BuildHomotopyTree.jl")
include("services/GaussianProductUtils.jl")
include("services/HomotopyBeliefPropagation.jl")

# AMP types and some legacy support 
include("entities/ManifoldDefinitions.jl")
include("services/ManifoldPartials.jl")
include("Interface.jl")

# regular features
include("services/KernelEval.jl")
include("services/HomotopyDensity.jl")
include("KernelHilbertEmbeddings.jl")

include("TrackingLabels.jl")

# include("Serialization.jl") # moved downstream to IIF to use InferenceVariable serialized types instead
include("API.jl")

include("Deprecated.jl")

# weak dependencies
# include("../ext/WeakdepsPrototypes.jl")

end
