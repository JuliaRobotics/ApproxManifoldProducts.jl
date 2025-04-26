# tests for ApproxManifoldProducts.jl

using ApproxManifoldProducts
using Test

include("basics.jl")
include("ex_1D.jl")
include("ex_2D_rot.jl")
include("testManifoldConventions.jl")
include("testLieFundamentals.jl")
include("testManifoldPartial.jl")
include("testManiProductBigSmall.jl")
include("testBasicManiProduct.jl")
include("testMarginalProducts.jl")
include("testMMD.jl")
include("testPartialProductSE2.jl")
include("basic_se3.jl")
include("testSymmetry.jl")
include("testUpdateMKD.jl")
include("testMKDStats.jl")

# Manellic tree, (on-Manifold Ellipse Ball Tree)
include("manellic/testManifoldTreeConstr.jl")
include("manellic/testTreeEvaluation.jl")
include("manellic/testTreeEntropyBandwidth.jl")
include("manellic/testMultiscaleLabelSampling.jl")
include("manellic/testTreeKernelProducts.jl")

#
