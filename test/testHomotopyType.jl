
using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups, Manifolds
using StaticArrays, SparseArrays
# using PDMats
using LinearAlgebra


##

@defStateType(
  TestTranslation1, 
  TranslationGroup(1), 
  zeros(1)
)

##

# NOTE, this type is auto-generated -- not generally seen or used during nominal usage
hr = ApproxManifoldProducts.HomotopyRepr(;
  topologykind = BinaryTruncFixedDepth{3}(),
  formkind = ConcentratedGaussianKernel(),
  statekind = TestTranslation1(),
  partial = nothing,
)

@show string(hr)

@test_broken isconcretetype(hr)

hrdfg = HomotopyReprDFG(hr)

@test_broken isconcretetype(hrdfg)


## check non-default serialization type

hr = ApproxManifoldProducts.HomotopyRepr(;
  statekind = TranslationGroup(3), 
  partial = (1,3), 
)

@show string(hr)

@test_broken isconcretetype(hr)


##

points = [
    [-1.0],
    [3.0],
    [-2.0],
]

manif = TranslationGroup(1)
hr = HomotopyReprLive(
  BinaryTruncFixedDepth{3}(),
  ConcentratedGaussianKernel(),
  manif,
  nothing,
)

@test_throws ArgumentError HomotopyReprDFG(hr)

lknlT = ConcentratedGaussianKernel(points[1], [1;;]) |> typeof
# legacy leaf kernels
lkern = Vector{lknlT}(undef, length(points))    

##


hd = HomotopyDensity_legacy(
  TestTranslation1(),
  points,
  # trailing_forms,
)


HomotopyDensityDFG(hd)

##

