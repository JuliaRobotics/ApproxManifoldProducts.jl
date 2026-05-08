##

using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups
using JSON

import DistributedFactorGraphs: DFGJSONStyle, TypeMetadata, Packed, resolvePackedType
import StructUtils: StructUtils, structlike, lower, lift, make
import ApproxManifoldProducts: HomotopyRepr

# JSON.json("$varFolder/$(v.label).json", v; style = DFGJSONStyle())

@defStateType(
  TestTranslation2, 
  TranslationGroup(2), 
  zeros(2)
)

##

hr = HomotopyRepr{
  BinaryTruncFixedDepth{3},
  ConcentratedGaussianKernel, 
  TestTranslation2, 
  Nothing, 
}(TestTranslation2(), nothing)


##


# # StructUtils.structlike(::DFGJSONStyle, ::Type{<:HomotopyRepr}) = false
# # StructUtils.arraylike(::DFGJSONStyle, ::Type{<:HomotopyRepr}) = false
# # StructUtils.lower(::DFGJSONStyle, x::HomotopyRepr) = string(typeof(x))
# # function StructUtils.lift(::DFGJSONStyle, ::Type{<:HomotopyRepr}, x::AbstractString)
# #     return getfield(Main, :HomotopyRepr), nothing
# # end

# StructUtils.structlike(::Type{<:HomotopyRepr}) = false
# StructUtils.lower(T::HomotopyRepr) = lowerHomotopyRepr(T)
# StructUtils.lift(::Type{HomotopyRepr}, s) = liftHomotopyRepr(s)


# function lowerHomotopyRepr(varT::H) where {A,B,C,D, H<:HomotopyRepr{A,B,C,D}}
#   typemeta = TypeMetadata(typeof(varT))
#   partial = JSON.json(string(varT.partial); style = DFGJSONStyle())
#   return TypeMetadata(
#     typemeta.pkg,
#     Symbol(typemeta.name, "{", A,", ", B, ", ", C, ", ", partial, "}"),
#     typemeta.version,
#   )
# end


# function liftHomotopyRepr(type::DFG.JSON.Object)
#   @info type
#   error("FIXME liftHomotopyRepr not implemented yet")
# end

##


JSON.json("/tmp/hr.json", Packed(hr); style = DistributedFactorGraphs.DFGJSONStyle())

##


hr3 = JSON.parse("/tmp/hr.json"; style = DistributedFactorGraphs.DFGJSONStyle())


##


function demo_lowerpack(
  hode::HomotopyDensity
)
  Dict(
    :reprkind => string(typeof(hode.reprkind)),  # use the same techinque as DFG.statekindN
    :observability => hode.observability,
    :points => hode.points,                 # use same technique as DFG.statekind for points (possible hint from reprkind)
    :weights => hode.weights,
    :principal_coeffs => hode.principal_coeffs,
    :principal_elements => hode.principal_elements, # use same technique as DFG.statekind for points (possible hint from reprkind)
    :principal_forms => hode.principal_forms,   # likely cov matrices
    :trailing_forms => Dict(hode.trailing_forms.nzind .=> hode.trailing_forms.nzval), # likely diagonal covs
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
    principal_coeffs = jsondict["principal_coeffs"],
    principal_elements = jsondict["principal_elements"], # use same technique as DFG.statekind for points
    principal_forms = jsondict["principal_forms"],
    trailing_forms = sparsevec(jsondict["trailing_forms"]),
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





## DEV STUFF



using Test
using ApproxManifoldProducts
using DistributedFactorGraphs
using LieGroups
using JSON

import DistributedFactorGraphs: DFGJSONStyle, TypeMetadata, Packed, resolvePackedType
import StructUtils: StructUtils, structlike, lower, lift, make
import ApproxManifoldProducts: HomotopyRepr

# JSON.json("$varFolder/$(v.label).json", v; style = DFGJSONStyle())

@defStateType(
  TestTranslation2, 
  TranslationGroup(2), 
  zeros(2)
)

##

hr = HomotopyRepr{
  BinaryTruncFixedDepth{3},
  ConcentratedGaussianKernel, 
  TestTranslation2, 
  Nothing, 
}(TestTranslation2(), nothing)


struct MyType{T}
  t::Type{T}
end

function StructUtils.make(::Type{<:MyType}, x)
  error("THIS")
end
function liftMyType(type::JSON.Object)
  error("THAT")
end
StructUtils.lift(::DFGJSONStyle, ::Type{<:MyType}, s) = liftMyType(s)


StructUtils.structlike(::Type{<:MyType}) = false

##

a = MyType(Int)

jstr = JSON.json(Packed(a); style = DFGJSONStyle())

uo = JSON.parse(jstr; style = DFGJSONStyle())

# expecting uo to return the equivalent of MyType{Int}(t=Int) --- resolvePackedType only does type?

##