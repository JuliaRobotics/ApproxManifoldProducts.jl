

# # FIXME deprecate trunction naming
# const AbstractHomotopyTrunction = AbstractHomotopyTopology 

# FIXME, rename to BinaryTruncFixedDepth{3}
struct MajorMaxDepth{N} <: AbstractHomotopyTopology end
# struct PrincipalInjectivityThres{N} <: AbstractHomotopyTopology{N} end
# struct PrincipalEnergyThres{N} <: AbstractHomotopyTopology{N} end

# Binary tree with N major levels
getMajorsLength(::Type{MajorMaxDepth{N}}) where {N} = N^2 - 1



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
    principal_details::Vector{MJ} = Vector{Matrix{Float64}}(undef, getMajorsLength(reprkind))
    """
    Store minor details such as leaf bandwidth or eigenvectors associated with minor eigenvalues.
    - When lifted for compute efficiency, this field is likely to hold something like PDMats.
    - When lowered or for serde, this field is likely to hold Dict{Int, Vector{Float64}}.
    """
    trailing_details::SparseArrays.SparseVector{MI, Int} = SparseArrays.sparsevec(
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
  tdks = keys(hode.trailing_details)
  tdvs = values(hode.trailing_details)
  rc = 0 < length(tdvs) ? size(tdvs[1]) : (0, 0)
  trailing_details = SparseArrays.sparsevec(Dict{Int,_second(eltype(tdvs))}(
    tdks .=> SMatrix{rc...}.(tdvs)
  ))
  
  return HomotopyDensityLive(
    reprkind = hode.reprkind,
    observability = hode.observability,
    points = hode.points,
    weights = hode.weights,
    principal_coeffs = hode.principal_coeffs,   # FIXME, convert type
    principal_elements = hode.principal_elements, # FIXME, convert type 
    principal_details = hode.principal_details,  # FIXME, convert type
    trailing_details,
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
  principal_details = Matrix.(hode.principal_details),
  trailing_details = Dict(hode.trailing_details.nzind .=> Matrix.(hode.trailing_details.nzval)), # likely diagonal covs
  structure = Dict(hode.structure.nzind .=> hode.structure.nzval),
)

