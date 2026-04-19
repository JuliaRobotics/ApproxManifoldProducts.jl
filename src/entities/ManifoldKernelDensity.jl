


"""
$TYPEDEF

Hybrid belief representation with natural transition between (non)parametric representations.


Notes:
- Refactoring and renaming of ManellicTree + ManifoldKernelDensity
- Replaces kernel density estimate, Gaussian mixture models, Principle component analysis, Homotopy methods, Model order reduction
- Allows partials as identified by list of coordinate dimensions e.g. `._partial = [1;3]`
  - When building a partial belief, use full points with necessary information in the specified partial coords.

In model order reduction, PCA, and modal analysis, the terms for the eigenvectors associated with the largest and smallest eigenvalues are commonly:
Major eigenvectors are often called "dominant eigenvectors," or simply "leading modes." In Principal Component Analysis (PCA), these are the "principal components."
Minor eigenvectors are sometimes called "trailing eigenvectors," or "residual modes." In PCA, these correspond to the components with the smallest variance.
"""
@kwdef struct HomotopyDensity{
  partial,
  M, 
  D <: AbstractVector,
  N, 
  HL, 
  HT,
  # U <: Union{<:StaticArray, <:PDMats.AbstractPDMat}
}
    manifold::M
    data::D
    weights::MVector{N, <:Real} = MVector{length(data), Float64}(ones(length(data))) ./ length(data)  # TODO rename to mixture_weights
    permute::MVector{N, <:Int} = MVector{length(data), Int}(1:length(data))
    leaf_kernels::SizedVector{N, HL}  # TODO rename to trailing
    tree_kernels::SizedVector{N, HT}  # TODO rename to leading
    segments::SizedVector{N, Set{Int}} = SizedVector{length(data), Set{Int}}(undef)
    infoPerCoord::Vector{Float64} = zeros(manifold_dimension(manifold))
    # _unibw::U

    # workaround to overcome bug for StaticArrays `isdefined() != false` issue || 
    #  use isassigned(), but recall same issue remains
    #  also will be influenced by serialization design, see #315
    _workaround_isdef_leafkernel::Set{Int} = Set{Int}()
    _workaround_isdef_treekernel::Set{Int} = Set{Int}()
end

HomotopyDensity{
  partial
}(;
  manifold::M, 
  data::D,
  leaf_kernels::SizedVector{N,HL},
  tree_kernels::SizedVector{N,HT},
  kw...
) where {partial, M, D, N, HL, HT} = 
HomotopyDensity{partial, M, D, length(data), HL, HT}(;
  manifold,
  data,
  leaf_kernels,
  tree_kernels,
  kw...
)


"""
    $TYPEDEF

On-manifold kernel density belief.

Notes
- Allows partials as identified by list of coordinate dimensions e.g. `._partial = [1;3]`
  - When building a partial belief, use full points with necessary information in the specified partial coords.

DevNotes
- WIP AMP issue 41, use generic retractions during manifold products.
"""
struct ManifoldKernelDensity{M <: MB.AbstractManifold, B <: HomotopyDensity, L}
    manifold::M
    """ legacy expects matrix of coordinates (as columns) """
    belief::B
    _partial::L
    """ just an example point for local access to the point data type"""
    # _u0::P
    # infoPerCoord::Vector{Float64}
end


