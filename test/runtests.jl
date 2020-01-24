# tests for ApproxManifoldProducts.jl

using ApproxManifoldProducts
using Test

include("basics.jl")

include("ex_1D.jl")

include("ex_2D_rot.jl")

include("testManiProductBigSmall.jl")

include("testBasicManiProduct.jl")

include("testMMD.jl")

#
