


abstract type AbstractBinaryTreeDensity <: AbstractHomotopyTopology end
const BinaryTreeDensity = AbstractBinaryTreeDensity

struct BinaryTruncFixedDepth{N} <: AbstractBinaryTreeDensity end
# struct BinaryInjectivityThres{N} <: AbstractBinaryTreeDensity{N} end
# struct PrincipalEnergyThres{N} <: AbstractHomotopyTopology{N} end



struct HomotopyReprLive{
  T <: Union{<:StateType, <:AbstractManifold},
  # Nothing and Tuple are legacy before DFG v1.0, trying to avoid serde of old partials devoid of serde design
  L <: Union{<:AbstractPartialTrait, Nothing, <:Tuple}, 
}
  topologykind::AbstractHomotopyTopology  # bitmap, jpeg, png
  reprkind::AbstractDensityForm           # RGB24, YCbCr, fullcov, uppercov, LieExpGaussianWrappedKind, ConcentrGaussKernelKind
  statekind::T                            # Position{2}
  partial::L
end

const HomotopyRepr = Union{<:HomotopyReprDFG, <:HomotopyReprLive}

function HomotopyRepr(
  repr::HomotopyRepr = HomotopyReprLive(
    BinaryTruncFixedDepth{3}(),
    ConcentratedGaussianKernel(),
    TranslationGroup(1),
    nothing,    
  );
  topologykind::AbstractHomotopyTopology = getTopologyKind(repr),
  reprkind::AbstractDensityForm = getReprKind(repr),
  statekind::Union{<:AbstractStateType, <:AbstractManifold} = getStateKind(repr),
  partial = getPartial(repr), # FIXME
) 
  return HomotopyReprLive(
    topologykind,
    reprkind,
    statekind,
    partial,
  )
end




@kwdef struct HomotopyDensityLive{
  H <: HomotopyRepr, # serde friendly when using DFG.statekind representation, but also supports Manifolds.jl direclty
  # parameters below auto generate during JSON lift, only above needs to be serde friendly
  P, # serde relies on DFG statekind mechanism, does not guarantee serde when directly using Manifolds wo DFG.statekind
  ME,  # Major elements can be points or eigen vectors etc.
  MJ,  # Use only easy to JSON.jl lift lower serde -- e.g. Dict{Int, Vector{Float64}} when storing just diagonal covariances for leaves of tree, or similar
  MI,  # Use only easy to JSON.jl lift lower serde -- e.g. Dict{Int, Vector{Float64}} when storing just diagonal covariances for leaves of tree, or similar
}
  reprkind::H
  observability::Vector{Float64} = zeros(manifold_dimension(getManifold(reprkind)))
  points::Vector{P}
  weights::Vector{Float64} = Vector{Float64}(ones(length(points))) ./ length(points)
  principal_coeffs::Vector{Float64} = Vector{Float64}(undef, getMajorsLength(reprkind))
  # FIXME, future proof such that ME != P, possibly using affine_matrix
  principal_elements::Vector{ME} = Vector{P}(undef, getMajorsLength(reprkind))
  principal_forms::Vector{MJ} = Vector{Matrix{Float64}}(undef, getMajorsLength(reprkind))
  """
  Store minor details such as leaf bandwidth or eigenvectors associated with minor eigenvalues.
  - When lifted for compute efficiency, this field is likely to hold something like PDMats.
  - When lowered or for serde, this field is likely to hold Dict{Int, Vector{Float64}}.
  """
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
    structure = SparseArrays.sparsevec(hode.structure),
  )
end
HomotopyDensityDFG(hode::HomotopyDensityLive) = HomotopyDensityDFG(
  reprkind = hode.reprkind,
  observability = hode.observability,
  points = hode.points,
  weights = hode.weights,
  principal_coeffs = hode.principal_coeffs,
  principal_elements = hode.principal_elements,
  principal_forms = Matrix.(hode.principal_forms),
  trailing_forms = Dict(hode.trailing_forms.nzind .=> Matrix.(hode.trailing_forms.nzval)), # likely diagonal covs
  structure = Dict(hode.structure.nzind .=> hode.structure.nzval),
)






getTopologyKind(reprkind::HomotopyRepr) = reprkind.topologykind
getTopologyKind(hode::HomotopyDensity) = getTopologyKind(hode.reprkind)
# getTopologyKind(state::State) = getTopologyKind(state.belief)

getReprKind(repr::HomotopyRepr) = repr.reprkind
getReprKind(hode::HomotopyDensity) = getReprKind(hode.reprkind)
# getReprKind(state::State) = getReprKind(state.belief)

getStateKind(repr::HomotopyRepr) = repr.statekind
getStateKind(hode::HomotopyDensity) = getStateKind(hode.reprkind)

getManifold(reprkind::HomotopyRepr) = getManifold(reprkind.statekind)
# getManifold(reprkind::HomotopyReprDFG) = getManifold(reprkind.statekind)
getManifold(hode::HomotopyDensity) = getManifold(hode.reprkind)
# getManifold(state::State) = getManifold(state.belief)

getPartial(repr::HomotopyReprLive) = repr.partial
getPartial(repr::HomotopyDensity) = getPartial(repr.reprkind)

_vanillareprT(::T) where {T <:ConcentratedGaussianKernel} = ConcentratedGaussianKernel

# Binary tree with N major levels
getMajorsLength(::Type{BinaryTruncFixedDepth{N}}) where {N} = N^2 - 1
getMajorsLength(kind::AbstractHomotopyTopology) = getMajorsLength(typeof(kind))
getMajorsLength(repr::HomotopyRepr) = getMajorsLength(repr.topologykind)
