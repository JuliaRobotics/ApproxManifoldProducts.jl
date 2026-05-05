# tests for ApproxManifoldProducts.jl

using ApproxManifoldProducts
using LieGroups
using Manifolds
using Test


##

include("testHomotopyType.jl")

include("testLieFundamentals.jl")
include("testManifoldConventions.jl")
# Manellic tree, (on-Manifold Ellipse Ball Tree)
include("manellic/testManifoldTreeConstr.jl")
include("manellic/testTreeEvaluation.jl")
include("testSymmetry.jl")
include("testUpdateMKD.jl")
include("testMKDStats.jl")
include("testMMD.jl")
include("basic_se3.jl")
include("manellic/testTreeEntropyBandwidth.jl")
include("testManifoldPartial.jl")
# density products
include("manellic/testTreeKernelProducts.jl")
include("manellic/testMultiscaleLabelSampling.jl")
include("testBasicManiProduct.jl")
include("testManiProductBigSmall.jl")
# tests for partials and partial products
include("testMarginalProducts.jl")
@error "testPartialProductSE2.jl is currently broken, TODO after HomotopyDensity refactor"
# include("testPartialProductSE2.jl")



#
