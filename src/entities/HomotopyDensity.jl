


"""
$TYPEDEF

Hybrid belief representation with natural transition between (non)parametric representations.


Notes:
- Refactoring and renaming of ManellicTree + ManifoldKernelDensity
- Replaces kernel density estimate, Gaussian mixture models, Principle component analysis, Homotopy methods, Model order reduction
- Allows partials as identified by list of coordinate dimensions e.g. `partial = [1;3]`
  - When building a partial belief, use full points with necessary information in the specified partial coords.

In model order reduction, PCA, and modal analysis, the terms for the eigenvectors associated with the largest and smallest eigenvalues are commonly:
Major eigenvectors are often called "dominant eigenvectors," or simply "leading modes." In Principal Component Analysis (PCA), these are the "principal components."
Minor eigenvectors are sometimes called "trailing eigenvectors," or "residual modes." In PCA, these correspond to the components with the smallest variance.
"""
@kwdef struct HomotopyDensity{
  partial,
  M, 
  P <: AbstractArray,
  HL, 
  HT,
  H <: HomotopyRepresentation
}
    representationkind::H
    manifold::M
    data::Vector{P}
    weights::Vector{Float64} = Vector{Float64}(ones(length(data))) ./ length(data)  # TODO rename to mixture_weights
    """ 
    Geometric data permute field, allows fast binary tree operations and geometric data splits for manellic (ball) trees. 
    - Geometric split reqs at most 2*(N+1)-1 elements -- e.g. when nodes have only right children, data=[1,2,-3].
    """
    geometric_permute::SparseArrays.SparseVector{Vector{Int}, Int} = SparseArrays.sparsevec(
      Dict(1 => collect(1:length(data))), 
      5*(length(data)) # large buffer space where impact on resources mitigated via sparsevec
    )
    leaf_kernels::Vector{HL}  # TODO rename to trailing
    tree_kernels::Vector{HT}  # TODO rename to leading
    infoPerCoord::Vector{Float64} = zeros(manifold_dimension(manifold))
end







