
using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups, Manifolds



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
  nothing, 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
}

@show string(hr)

@test isconcretetype(hr)


## check non-default serialization type

hr = ApproxManifoldProducts.HomotopyRepresentation{
  typeof(TranslationGroup(3)), 
  (1,3), 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
}

@show string(hr)

@test isconcretetype(hr)


##



pts = [
    [-1.0],
    [3.0],
    [-2.0],
]

manif = TranslationGroup(1)
hr = ApproxManifoldProducts.HomotopyRepresentation{
  typeof(manif), 
  nothing, 
  ConcentratedGaussianKernel, 
  MajorMaxDepth{3}
}(manif)



lknlT = ConcentratedGaussianKernel(pts[1], [1;;]) |> typeof
# leaf kernels
lkern = Vector{lknlT}(undef, length(pts))

##

HomotopyDensity{
  typeof(hr),
  eltype(pts),
  lknlT,
  lknlT,
}(;
  representationkind = hr,
  elements = pts,
  # TODO deprecating fields below
  leaf_kernels = lkern,
  tree_kernels = lkern,
)






##