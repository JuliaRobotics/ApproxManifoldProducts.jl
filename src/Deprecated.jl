

## ======================================================================================================
## Remove below before v0.14
## ======================================================================================================


# const MKD{M, B, L} = ManifoldKernelDensity{M, B, L}

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

# # ManellicTree

# # Short for Manifold Ellipse Metric Tree
# # starting as a balanced tree, relax to unbalanced in future.
# struct ManellicTree{
#   M, 
#   D <: AbstractVector, 
#   N, 
#   HL, 
#   HT
# }
#     manifold::M
#     data::D
#     weights::MVector{N, <:Real}  # TODO rename to mixture_weights
#     # TODO introduce weights
#     permute::MVector{N, Int}
#     # kernels::ArrayPartition{<:Number,KT}
#     """ These leaf kernels store a duplication of .data[.permute] """
#     leaf_kernels::SizedVector{N, HL}  # TODO rename to twig_kernels
#     tree_kernels::SizedVector{N, HT}  # TODO rename to mixture_kernels
#     segments::SizedVector{N, Set{Int}}
#     # left_idx::MVector{N,Int}
#     # right_idx::MVector{N,Int}

#     # workaround to overcome bug for StaticArrays `isdefined() != false` issue
#     _workaround_isdef_leafkernel::Set{Int}
#     _workaround_isdef_treekernel::Set{Int}
# end


# function Base.getproperty(mt::ManellicTree{M,D,N},f::Symbol) where {M,D,N}
#   if f !== :kernel
#     getfield(mt, f)
#   else

#   end
# end

# const ManifoldKernelDensity{M, D, L, N} = HomotopyDensity{L, M, D, N}

# # helper constructor for common case without partials
# ConcentratedGaussianKernel{partial}(;
#     weight::Float64,
#     functional::K,
#     params::T, 
#     # partial::P = nothing,
# ) where {
#     partial,
#     K,
#     T
# } = ConcentratedGaussianKernel{
#     partial,
#     K,
#     T
# }(;
#     weight,
#     functional,
#     params
# )

# ConcentratedGaussianKernel(
#     p=SVector(0.0),  # center/expansion point on the manifold
#     covmat=SMatrix{1,1}(1.0);
#     weight::Number=1.0,
#     partial::P = nothing,
# ) where P <: Union{Nothing, <:Tuple} = ConcentratedGaussianKernel{partial}(;
#     weight, 
#     functional=MvNormal(covmat), # NOTE, find inverse Cholesky in MvNormal structure
#     params=p,
# )

# struct MvNormalKernel{T <: DensityKernel} <: AbstractKernel
#     shim::T
# end

# # case for identical types not requiring any conversions
# function Base.convert(
#     ::Type{T},
#     src::T,
# ) where {T <: MvNormalKernel}
#     return src
# end

# function distanceMalahanobisSq(
#     M::AbstractLieGroup,
#     K::AbstractKernel,
#     q,
#     basis=DefaultOrthogonalBasis();
#     partial::Union{Nothing,<:Tuple} = nothing,
# )
#     δc = distanceMalahanobisCoordinates(M, K, q)
#     # return inner(M, p, X, X) # did not work as inner gave almost 2x the answer?
#     return δc' * δc
# end

# import Base: getproperty
# function Base.getproperty(k::MvNormalKernel, f::Symbol)
#     if f === :sqrt_iΣ
#         # super slow and hacky, but only legacy.  WIP replacing
#         cov(k) |> inv |> sqrt
#     else
#         return getproperty(k.shim, f)
#     end
# end

## ======================================================================================================
## Remove below before v0.13
## ======================================================================================================

