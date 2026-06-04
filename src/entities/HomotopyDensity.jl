


abstract type AbstractBinaryTreeDensity <: AbstractHomotopyTopology end
const BinaryTreeDensity = AbstractBinaryTreeDensity

struct BinaryTruncFixedDepth{N} <: AbstractBinaryTreeDensity end
# struct BinaryInjectivityThres{N} <: AbstractBinaryTreeDensity{N} end
# struct PrincipalEnergyThres{N} <: AbstractHomotopyTopology{N} end

struct PartialNoSerde <: AbstractPartialTrait end


struct HomotopyReprLive{
  T <: Union{<:StateType, <:AbstractManifold},
  # Nothing and Tuple are legacy before DFG v1.0, trying to avoid serde of old partials devoid of serde design
  L <: Union{<:AbstractPartialTrait, Nothing, <:Tuple}, 
}
  topologykind::AbstractHomotopyTopology  # bitmap, jpeg, png
  formkind::AbstractDensityForm           # RGB24, YCbCr, fullcov, uppercov, LieExpGaussianWrappedKind, ConcentrGaussKernelKind
  statekind::T                            # Position{2}
  partial::L
end

const HomotopyRepr = Union{<:HomotopyReprLive, <:HomotopyReprDFG}

function HomotopyRepr(
  repr::HomotopyRepr = HomotopyReprLive(
    BinaryTruncFixedDepth{3}(),
    ConcentratedGaussianKernel(),
    TranslationGroup(1),
    nothing,    
  );
  topologykind::AbstractHomotopyTopology = getTopologyKind(repr),
  formkind::AbstractDensityForm = getFormKind(repr),
  statekind::Union{<:AbstractStateType, <:AbstractManifold} = getStateKind(repr),
  partial = getPartial(repr), # FIXME
) 
  return HomotopyReprLive(
    topologykind,
    formkind,
    statekind,
    partial,
  )
end


function HomotopyReprDFG(
  hr::HomotopyReprLive;
  topologykind = hr.topologykind,
  formkind = hr.formkind,
  statekind = hr.statekind,
  partial = hr.partial,
)
  if !(statekind isa AbstractStateType)
    throw(ArgumentError("JSON.jl serde using DFG representation requires statekind to be a StateType -- you can easily expand serde support for your manifold with DistributedFactorGraph.@defStateType.  Alternatively, Homotopy*Live supports direct use of Manifolds.jl types without serde: $statekind"))
  end
  if !(partial isa Nothing)
    # throw(ArgumentError("JSON.jl serde using DFG representation does not currently support partials -- you can easily expand serde support for your partial with DistributedFactorGraph.@defPartialTrait.  Alternatively, Homotopy*Live supports direct use of partials without serde."))
    @error("JSON.jl serde using DFG representation does not currently support partials -- you can easily expand serde support for your partial with DistributedFactorGraph.@defPartialTrait.  Alternatively, Homotopy*Live supports direct use of partials without serde.")
  end
  return DistributedFactorGraphs.HomotopyReprDFG(
    topologykind,
    formkind,
    statekind,
    PartialNoSerde(), # TODO
  )
end

function HomotopyReprLive(
  hr::HomotopyReprDFG;
  topologykind = hr.topologykind,
  formkind = hr.formkind,
  statekind = hr.statekind,
  partial = nothing, # TODO
)
  return HomotopyReprLive(
    topologykind,
    formkind,
    statekind,
    partial,
  )
end


convert(::Type{<:HomotopyReprLive}, src::HomotopyReprDFG) = HomotopyReprLive(src)
convert(::Type{<:HomotopyReprDFG}, src::HomotopyReprLive) = HomotopyReprDFG(src)



@kwdef struct HomotopyDensityLive{
  H <: HomotopyRepr, # serde friendly when using DFG.statekind representation, but also supports Manifolds.jl direclty
  # parameters below auto generate during JSON lift, only above needs to be serde friendly
  P, # serde relies on DFG statekind mechanism, does not guarantee serde when directly using Manifolds wo DFG.statekind
  ME,  # Major elements can be points or eigen vectors etc.
  MJ,
  MI,
}
  reprkind::H
  observability::Vector{Float64} = zeros(manifold_dimension(getManifold(reprkind)))
  principal_coeffs::Vector{Float64} = Vector{Float64}(undef, getMajorsLength(reprkind))
  # FIXME, future proof such that ME != P, possibly using affine_matrix
  principal_elements::Vector{ME} = Vector{P}(undef, getMajorsLength(reprkind))
  principal_forms::Vector{MJ} = Vector{Matrix{Float64}}(undef, getMajorsLength(reprkind))
  """
  Store minor details such as leaf bandwidth or eigenvectors associated with minor eigenvalues.
  - When lifted for compute efficiency, this field is likely to hold something like PDMats.
  - When lowered or for serde, this field is likely to hold Dict{Int, Vector{Float64}}.
  """  
  points::Vector{P}
  weights::Vector{Float64} = Vector{Float64}(ones(length(points))) ./ length(points)
  trailing_forms::SparseArrays.SparseVector{MI, Int} = SparseArrays.sparsevec(
    Dict(1 => SMatrix{Float64}(I, manifold_dimension(getManifold(reprkind)), manifold_dimension(getManifold(reprkind))),),
    1
  )
  """ 
  Geometric points permute field, allows fast binary tree operations and geometric points splits for manellic (ball) trees. 
  - Geometric split reqs at least 2*(N+1)-1 points -- e.g. when nodes have only right children, points=[1,2,-3].
  """
  structure::SparseArrays.SparseVector{Vector{Int}, Int} = SparseArrays.sparsevec(
    Dict(1 => collect(1:length(points))), 
    5*(length(points)) # large buffer space where impact on resources mitigated via sparsevec
  )
end


## consolidate in IIF had weird cases requiring conversion between MArray and SArray types.  This convert became the easiest
# trivial case with no convert between the same P point types
convert(::Type{<:HomotopyDensityDFG{S,P}}, src::HomotopyDensityDFG{S,P}) where {S,P} = src
function convert(
  ::Type{<:HomotopyDensityDFG{S,sP}}, 
  src::HomotopyDensityDFG{S,P}
) where {S, sP, P}
  # @info "convert" string(src.principal_elements) string(src.points) string(src.trailing_forms)

  principal_elements = Vector{sP}(undef, length(src.principal_elements))
  for i in 1:length(src.principal_elements)
    if isassigned(src.principal_elements, i)
      principal_elements[i] = convert(sP, src.principal_elements[i])
    end
  end
  points = Vector{sP}(undef, length(src.points))
  for i in 1:length(src.points)
    if isassigned(src.points, i)
      points[i] = convert(sP, src.points[i])
    end
  end

  return HomotopyDensityDFG{S, sP}(;
    reprkind = src.reprkind,
    observability = src.observability,
    principal_coeffs = src.principal_coeffs,
    principal_elements,
    principal_forms = src.principal_forms,
    points,
    weights = src.weights,
    trailing_forms = src.trailing_forms,
    structure = src.structure,
  )
end


# Solve DataLevel 3.5
const HomotopyDensity = Union{<:DistributedFactorGraphs.HomotopyDensityDFG, <:HomotopyDensityLive}

# overload Base.convert for easy conversion between live and hold representations
convert(::Type{<:HomotopyDensityLive}, src::HomotopyDensityDFG) = HomotopyDensityLive(src)
convert(::Type{<:HomotopyDensityDFG}, src::HomotopyDensityLive) = HomotopyDensityDFG(src)



function HomotopyDensityLive(hode::HomotopyDensityDFG)
  _second(::Pair{K, V}) where {K, V} = V
  # convert trailing details from Dict{Int, Matrix{Float64}} to SparseVector{Int, SMatrix{N,N}} for live representation
  tdks = keys(hode.trailing_forms)
  tdvs = values(hode.trailing_forms)
  rc = 0 < length(tdvs) ? size(tdvs[1]) : (0, 0)
  trailing_forms = SparseArrays.sparsevec(Dict{Int,_second(eltype(tdvs))}(
    tdks .=> SMatrix{rc...}.(tdvs)
  ))
  
  return HomotopyDensityLive(
    reprkind = hode.reprkind,
    observability = hode.observability,
    points = hode.points,
    weights = hode.weights,
    principal_coeffs = hode.principal_coeffs,   # FIXME, convert type
    principal_elements = hode.principal_elements, # FIXME, convert type 
    principal_forms = hode.principal_forms,  # FIXME, convert type
    trailing_forms,
    structure = SparseArrays.sparsevec(hode.structure.nzidx, hode.structure.nzval),
  )
end
HomotopyDensityDFG(hode::HomotopyDensityLive) = HomotopyDensityDFG(
  reprkind = HomotopyReprDFG(hode.reprkind),
  observability = hode.observability,
  points = hode.points,
  weights = hode.weights,
  principal_coeffs = hode.principal_coeffs,
  principal_elements = hode.principal_elements,
  principal_forms = begin
    pf = Vector{Matrix{Float64}}(undef, length(hode.principal_forms))
    for i in 1:length(hode.principal_forms) 
      if isassigned(hode.principal_forms, i)
        pf[i] = Matrix(hode.principal_forms[i])
      end
    end 
    pf
  end,
  trailing_forms = SparseArrays.sparsevec(hode.trailing_forms.nzind, Matrix.(hode.trailing_forms.nzval)), # likely diagonal covs
  structure = hode.structure,
)



getTopologyKind(reprkind::HomotopyRepr) = reprkind.topologykind
getTopologyKind(hode::HomotopyDensity) = getTopologyKind(hode.reprkind)

getFormKind(repr::HomotopyRepr) = repr.formkind
getFormKind(hode::HomotopyDensity) = getFormKind(hode.reprkind)

getStateKind(repr::HomotopyRepr) = repr.statekind
getStateKind(hode::HomotopyDensity) = getStateKind(hode.reprkind)

getManifold(reprkind::HomotopyRepr) = getManifold(reprkind.statekind)
# getManifold(reprkind::HomotopyReprDFG) = getManifold(reprkind.statekind)
# getManifold(hode::HomotopyDensity) = getManifold(hode.reprkind)
# getManifold(state::State) = getManifold(state.belief)

getPointType(x::HomotopyDensity) = eltype(x.points) # TODO use HomotopyDensity{T} style instead
function getManifold(hode::HomotopyDensity, aspartial::Bool = false)
    return if !aspartial
        getManifold(hode.reprkind)
    else
        M_, _, _ = getManifoldPartial(getManifold(hode), getPartial(hode), hode.points[1])
        M_
    end
end


getPartial(repr::HomotopyRepr) = repr.partial
getPartial(hode::HomotopyDensity) = getPartial(hode.reprkind)

_vanillareprT(::T) where {T <:ConcentratedGaussianKernel} = ConcentratedGaussianKernel

# Binary tree with N major levels
getMajorsLength(::Type{BinaryTruncFixedDepth{N}}) where {N} = N^2 - 1
getMajorsLength(kind::AbstractHomotopyTopology) = getMajorsLength(typeof(kind))
getMajorsLength(repr::HomotopyRepr) = getMajorsLength(repr.topologykind)
