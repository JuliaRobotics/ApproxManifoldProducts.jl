
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
hr = ApproxManifoldProducts.HomotopyRepr{
  BinaryTruncFixedDepth{3},
  ConcentratedGaussianKernel, 
  TestTranslation1, 
  Nothing, 
}

@show string(hr)

@test isconcretetype(hr)


## check non-default serialization type

hr = ApproxManifoldProducts.HomotopyRepr{
  BinaryTruncFixedDepth{3},
  ConcentratedGaussianKernel, 
  typeof(TranslationGroup(3)), 
  Tuple{Int,Int}, 
}

@show string(hr)

@test isconcretetype(hr)


##

points = [
    [-1.0],
    [3.0],
    [-2.0],
]

manif = TranslationGroup(1)
hr = ApproxManifoldProducts.HomotopyRepr{
  BinaryTruncFixedDepth{3},
  ConcentratedGaussianKernel, 
  typeof(manif), 
  Nothing, 
}(manif, nothing)



lknlT = ConcentratedGaussianKernel(points[1], [1;;]) |> typeof
# legacy leaf kernels
lkern = Vector{lknlT}(undef, length(points))

##

# d = manifold_dimension(getManifold(hr))
# trailing_forms = SparseArrays.sparsevec(Dict(
#   1 => SMatrix{d,d}(I),
# ), 1)


hd = HomotopyDensity_legacy(
  manif,
  points,
  # trailing_forms,
)



##