


# """
#     $TYPEDEF

# Elliptical structure for use in a (Manellic) Ball Tree.
# """
# struct HyperEllipse{P <:AbstractArray,D,DD}
#   """ manifold point at which this ellipse is based """
#   point::P
#   """ Covariance of coords at either TBD this point or some other reference point? """
#   coord_cov::SMatrix{D,D,Float64,DD}
# end

# ManellicTree

# Short for Manifold Ellipse Metric Tree
# starting as a balanced tree, relax to unbalanced in future.
struct ManellicTree{M,D<:AbstractVector,N,HL,HT}
  manifold::M
  data::D
  weights::MVector{N,<:Real}
  permute::MVector{N,Int}
  # kernels::ArrayPartition{<:Number,KT}
  leaf_kernels::SizedVector{N,HL}
  tree_kernels::SizedVector{N,HT}
  segments::SizedVector{N,Set{Int}}
  # left_idx::MVector{N,Int}
  # right_idx::MVector{N,Int}

  # workaround to overcome bug for StaticArrays `isdefined() != false` issue
  _workaround_isdef_leafkernel::Set{Int}
  _workaround_isdef_treekernel::Set{Int}
end

