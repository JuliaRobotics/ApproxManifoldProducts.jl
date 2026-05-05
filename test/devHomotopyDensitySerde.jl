##

using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups


@defStateType(
  TestTranslation2, 
  TranslationGroup(2), 
  zeros(2)
)

##

function demo_lowerpack(
  hode::HomotopyDensity
)
  Dict(
    :reprkind => string(typeof(hode.reprkind)),  # use the same techinque as DFG.statekindN
    :observability => hode.observability,
    :points => hode.points,                 # use same technique as DFG.statekind for points (possible hint from reprkind)
    :weights => hode.weights,
    :majors_coeff => hode.majors_coeff,
    :majors_element => hode.majors_element, # use same technique as DFG.statekind for points (possible hint from reprkind)
    :majors_detail => hode.majors_detail,   # likely cov matrices
    :minors_detail => Dict(hode.minors_detail.nzind .=> hode.minors_detail.nzval), # likely diagonal covs
    :structure => Dict(hode.structure.nzind .=> hode.structure.nzval),
  )
end


function demo_liftunpack(
  jsondict::Dict{String, Any}
)
  HomotopyDensity(;
    reprkind = getfield(Main, jsondict["reprkind"]), # use the same technique as DFG.statekindN
    observability = jsondict["observability"],
    points = jsondict["points"],                 # use same technique as DFG.statekind for points
    weights = jsondict["weights"],
    majors_coeff = jsondict["majors_coeff"],
    majors_element = jsondict["majors_element"], # use same technique as DFG.statekind for points
    majors_detail = jsondict["majors_detail"],
    minors_detail = sparsevec(jsondict["minors_detail"]),
    structure = sparsevec(jsondict["structure"]),
  )
end


## some example data -- classic unimodal density in x and y.

hode = HomotopyDensity_legacy(
  TestTranslation2(), # LieGroups.TranslationGroup(2) can compute but not DFG-lazyserde compatible 
  [randn(2),];
  bw = [0.25;0.5]
)

##

rootkernel = getKernelTree(hode, 1)

@show mean(rootkernel)
@show cov(rootkernel)


## some example data -- conventional non-parametric.

hode = HomotopyDensity_legacy(
  TestTranslation2(), # LieGroups.TranslationGroup(2) can compute but not DFG-lazyserde compatible 
  [randn(2) for _ in 1:128];
)

##

rootkernel = getKernelTree(hode, 1)

@show mean(rootkernel)
@show cov(rootkernel)

##

