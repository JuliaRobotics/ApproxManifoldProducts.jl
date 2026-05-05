
using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups, Manifolds
using StaticArrays, SparseArrays
using PDMats
using LinearAlgebra


##

@defStateType(
  TestTranslation1, 
  TranslationGroup(1), 
  zeros(1)
)

##

# NOTE, this type is auto-generated -- not generally seen or used during nominal usage
hr = ApproxManifoldProducts.HomotopyRepresentation{
  TestTranslation1, 
  Nothing, 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
}

@show string(hr)

@test isconcretetype(hr)


## check non-default serialization type

hr = ApproxManifoldProducts.HomotopyRepresentation{
  typeof(TranslationGroup(3)), 
  Tuple{Int,Int}, 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
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
hr = ApproxManifoldProducts.HomotopyRepresentation{
  typeof(manif), 
  Nothing, 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
}(manif, nothing)



lknlT = ConcentratedGaussianKernel(points[1], [1;;]) |> typeof
# legacy leaf kernels
lkern = Vector{lknlT}(undef, length(points))

##

d = manifold_dimension(getManifold(hr))
minors_detail = SparseArrays.sparsevec(Dict(
  1 => PDMat(
    SMatrix{d,d}(I)
    ),
), 1)


hd = HomotopyDensity{
  typeof(hr),
  eltype(points),
  # lknlT,
  eltype(points),
  Matrix{Float64},
  eltype(minors_detail)
}(;
  reprkind = hr,
  points,
  minors_detail,
)



##