


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
    majors_coeff::Vector{Float64} = Vector{Float64}(undef, getMajorsLength(reprkind))
    majors_element::Vector{ME} = Vector{P}(undef, getMajorsLength(reprkind))
    majors_detail::Vector{MJ} = Vector{Matrix{Float64}}(undef, getMajorsLength(reprkind))
    """
    Store minor details such as leaf bandwidth or eigenvectors associated with minor eigenvalues.
    - When lifted for compute efficiency, this field is likely to hold something like PDMats.
    - When lowered or for serde, this field is likely to hold Dict{Int, Vector{Float64}}.
    """
    minors_detail::SparseArrays.SparseVector{MI, Int} = SparseArrays.sparsevec(
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




