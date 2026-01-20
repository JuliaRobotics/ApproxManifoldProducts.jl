
const TreeDensity = Union{<:ManellicTree, <:BallTreeDensity}

"""
    $TYPEDEF

On-manifold kernel density belief.

Notes
- Allows partials as identified by list of coordinate dimensions e.g. `._partial = [1;3]`
  - When building a partial belief, use full points with necessary information in the specified partial coords.

DevNotes
- WIP AMP issue 41, use generic retractions during manifold products.
"""
struct ManifoldKernelDensity{M <: MB.AbstractManifold, B <: TreeDensity, L, P}
    manifold::M
    """ legacy expects matrix of coordinates (as columns) """
    belief::B
    _partial::L
    """ just an example point for local access to the point data type"""
    _u0::P
    infoPerCoord::Vector{Float64}
end
# const MKD{M, B, L} = ManifoldKernelDensity{M, B, L}




"""
$TYPEDEF

Hybrid belief representation with natural transition between (non)parametric representations.


Notes:
- Refactoring and renaming of ManellicTree + ManifoldKernelDensity
- Kernel density estimate, Gaussian mixture models, Principle component analysis, Homotopy methods, Model order reduction

In model order reduction, PCA, and modal analysis, the terms for the eigenvectors associated with the largest and smallest eigenvalues are commonly:
Major eigenvectors are often called "dominant eigenvectors," or simply "leading modes." In Principal Component Analysis (PCA), these are the "principal components."
Minor eigenvectors are sometimes called "trailing eigenvectors," or "residual modes." In PCA, these correspond to the components with the smallest variance.

Also see: HomotopyBeliefKernel
"""
struct HomotopyBelief{
  M, 
  N, 
  D <: AbstractVector, 
  L,
  HL, 
  HT,
  U <: Union{<:StaticArray, <:PDMats.AbstractPDMat}
}
    manifold::M
    data::D
    weights::MVector{N, <:Real}  # TODO rename to mixture_weights
    permute::MVector{N, <:Int}
    leaf_kernels::SizedVector{N, HL}  # TODO rename to trailing
    tree_kernels::SizedVector{N, HT}  # TODO rename to leading
    segments::SizedVector{N, Set{Int}}
    infoPerCoord::Vector{Float64}
    _unibw::U
    _partial::L

    # workaround to overcome bug for StaticArrays `isdefined() != false` issue
    _workaround_isdef_leafkernel::Set{Int}
    _workaround_isdef_treekernel::Set{Int}
end

# const ManifoldKernelDensity{M, D, L, N} = HomotopyBelief{M, D, N, L}