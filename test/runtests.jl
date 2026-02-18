# tests for ApproxManifoldProducts.jl

using ApproxManifoldProducts
using LieGroups
using Manifolds
using Test

##

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
include("manellic/testTreeKernelProducts.jl")
include("manellic/testMultiscaleLabelSampling.jl")
include("testBasicManiProduct.jl")
include("testManiProductBigSmall.jl")
# include("testMarginalProducts.jl")
# include("testPartialProductSE2.jl")


## TODO update examples for new homotopyBelief interface
# include("ex_1D.jl")
# include("ex_2D_rot.jl")

# moved to attic
# include("basics.jl") # legacy KDE tests


#
