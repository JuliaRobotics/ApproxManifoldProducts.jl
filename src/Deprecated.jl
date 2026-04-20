
## ======================================================================================================
## Remove below before v0.15
## ======================================================================================================


@deprecate ManifoldKernelDensity(
  manifold::AbstractManifold,
  hode::HomotopyDensity,
  _partial::Union{Nothing, <:AbstractVector, <:Tuple},
  _u0::AbstractVector,
  infoPerCoord::AbstractVector{<:Real};
  partl_cb::Union{Nothing, <:Function} = nothing
) hode

# remove -- find workaround for partl_cb in this case
@deprecate ManifoldKernelDensity(
    hode::HomotopyDensity,
    ::Nothing;
    partl_cb::Nothing = nothing,
) hode

# Base.length(::HomotopyDensity{L, M, D, N}) where {L, M, D, N} = N


# function Base.show(io::IO, hode::HomotopyDensity{partial,M}) where {partial,M}
#     _round(s::AbstractArray; kw...) = round.(s[:]; kw...)
#     _round(s::AbstractVector{<:AbstractMatrix}; kw...) = round.(s[1][:]; kw...)

#     printstyled(io, "HomotopyDensity{"; bold = true, color = :blue)
#     println(io)
#     # FIXME restore after HomotopyDensity refactor
#     printstyled(io, "    partial"; bold = true, color = :magenta)
#     print(io, " = ", partial, ",")
#     println(io)
#     printstyled(io, "    M"; bold = true, color = :magenta)
#     print(io, " = ", M, ",")
#     println(io)
#     # printstyled(io, "    B"; bold = true, color = :magenta)
#     # print(io, " = ", B, ",")
#     # println(io)
#     # FIXME restore after HomotopyDensity refactor
#     println(io, " }(")
#     println(io, "  Npts:  ", Npts(hode))
#     print(io, "  dims:  ", Ndim(hode))
#     printstyled(io, isPartial(hode) ? "* --> $(length(getPartial(hode)))" : ""; bold = true)
#     println(io)
#     println(io, "  prtl:   ", getPartial(hode))
#     bw = (getBW(hode).^2)[:, 1]
#     pvec = isPartial(hode) ? getPartial(hode) : collect(1:length(bw))
#     println(io, "  bws:   ", getBandwidth(hode, true) |> x -> _round(x; digits = 4)) # .|> x->round(x,digits=4))
#     println(io, "  ipc:   ", getInfoPerCoord(hode, true) .|> x -> round(x; digits = 4))
#     print(io, "   mean: ")
#     try
#         mn = mean(hode)
#         if mn isa ProductRepr # TODO UPDATE to ArrayPartition only, discontinued use of ProductRepr long ago.
#             println(io)
#             for prt in mn.parts
#                 println(io, "         ", round.(prt, digits = 4))
#             end
#         else
#             println(io, round.(mn', digits = 4))
#         end
#     catch
#         println(io, "----")
#     end
#     println(io, ")")
#     return nothing
# end
# Base.show(io::IO, ::MIME"text/plain", mkd::HomotopyDensity) = show(io, mkd)


# getPoints(mt::HomotopyDensity; permute::Bool = true) = permute ? view(mt.data, mt.permute) : mt.data

# """
#     $SIGNATURES

# Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

# Notes
# - Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
# - Second argument controls whether partial dimensions only should be returned (`=true` default).

# DevNotes
# - Currently converts down to manifold from matrix of coordinates (legacy), to be deprecated TODO
# """
# function getPoints(
#     x::ManifoldKernelDensity,
#     aspartial::Bool = true;
#     permute::Bool = true,
# )
#     #
#     pts = getPoints(x.shim; permute)

#     if !isPartial(x) && !aspartial
#         error("MKD getPoints aspartial=true but MKD is not partial")
#         return pts
#     end

#     Mp, Rp, lkup = getManifoldPartial(getManifold(x), getPartial(x), pts[1])

#     vecP = Vector{typeof(Rp)}(undef, length(pts))
#     for (j,pt) in enumerate(pts)
#         vecP[j] =  lkup(pt)
#     end
#     return vecP
# end



# function getBW(
#     x::ManifoldKernelDensity{B},
#     asPartial::Bool = true;
#     kw...,
# ) where {B}
#     bws = getBW(x.shim; kw...)
#     if isPartial(x) && asPartial
#         return (bw->view(bw, getPartial(x))).(bws)
#     end
#     return bws
# end

# """
#     $TYPEDEF

# On-manifold kernel density belief.

# Notes
# - Allows partials as identified by list of coordinate dimensions e.g. `partial = [1;3]`
#   - When building a partial belief, use full points with necessary information in the specified partial coords.

# DevNotes
# - WIP AMP issue 41, use generic retractions during manifold products.
# """
# struct ManifoldKernelDensity{B <: HomotopyDensity}
#   # manifold::M
#   """ HomotopyDensity legacy-shim for hybrid-(non)parametric belief propagation """
#   shim::B
#   # _partial::L
#   # """ just an example point for local access to the point data type"""
#   # _u0::P
#   # infoPerCoord::Vector{Float64}
# end


# import Base: getproperty

# function getproperty(mkd::ManifoldKernelDensity, f::Symbol)
#   if f === :shim || f === :belief
#     return getfield(mkd, :shim)
#   elseif f === :manifold
#     return getManifold(mkd)
#   elseif f === :_partial
#     return getPartial(mkd)
#   elseif f === :_u0
#     return getPointRepr(mkd)
#   else
#     @warn "ManifoldKernelDensity has been deprecated, use HomotopyBelief instead." maxlog=20
#     getproperty(getfield(mkd, :shim), f)
#   end
# end


## ======================================================================================================
## Remove below before v0.14
## ======================================================================================================


# function getPoints(
#     x::ManifoldKernelDensity{B, Nothing},
#     ::Bool = true; # aspartial unused
#     permute::Bool = true,
# ) where {B}
#     return getPoints(x.belief; permute)
# end

# function _getFieldPartials(
#     mkd::ManifoldKernelDensity{B, Nothing},
#     field::Function,
#     _aspartial::Bool = true,
# ) where {B}
#     return field(mkd)
# end

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

