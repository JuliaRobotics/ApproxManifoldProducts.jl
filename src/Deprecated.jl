


# show requires this 
getPartial(repr::HomotopyReprDFG) = nothing # FIXME

## TYPE PIRACY!!!! FIXME -- upstream to DFG
getManifold(manif::AbstractManifold) = manif


## ======================================================================================================
## Remove below before v0.17
## ======================================================================================================
W

@deprecate HomotopyDensity(
  hode::HomotopyDensity;
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
) HomotopyDensity(hode, partial; observability = hode.observability)

# function HomotopyDensity(
#   hode::HomotopyDensity;
#   partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
# )
#   _workaround(::HomotopyDensityLive) = HomotopyDensityLive
#   _workaround(::HomotopyDensityDFG) = HomotopyDensityDFG
#   _HD = _workaround(hode) # FIXME remove after partial types are stable - i.e. drop Nothing vs Tuple

#   partl = getPartial(hode)
#   _partl = _intersect(partial, partl)
#   _HD(HomotopyDensity_legacy(
#     hode.reprkind, 
#     hode.points; 
#     partial=_partl,
#     principal_coeffs = hode.principal_coeffs,
#     principal_elements = hode.principal_elements,
#     principal_forms = hode.principal_forms,
#     weights = getWeights(hode),
#     structure = hode.structure,
#     observability = hode.observability, 
#     # kernel_bw, 
#     # kw...
#   ))
# #   HomotopyDensity_legacy(;
# #     partial = _partl,
# #     manifold = getManifold(hode),
# #     points = hode.points,
# #     principal_coeffs = hode.principal_coeffs,
# #     principal_elements = hode.principal_elements,
# #     principal_forms = hode.principal_forms,
# #     weights = getWeights(hode),
# #     structure = hode.structure,
# #     observability = hode.observability,
# #   )
# end

# function HomotopyDensity_legacy(;
#   partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
#   manifold::M, 
#   points::Vector{P},
#   kernel_bw = nothing,
#   kw...
# ) where {M, P}
#     _legacybw(s::AbstractMatrix) = s
#     _legacybw(s::AbstractVector) = diagm(s)
#     _legacybw(::Nothing) = LinearAlgebra.I
        
#     lCV = _legacybw(kernel_bw)

#     _partial = _tuple(partial)
#     reprkind = HomotopyRepr{
#         BinaryTruncFixedDepth{3},
#         ConcentratedGaussianKernel, 
#         M, 
#         typeof(_partial), 
#     }(manifold, _partial)

#     d = manifold_dimension(getManifold(reprkind))
#     trailing_forms = SparseArrays.sparsevec(Dict(
#         1 => SMatrix{d,d,Float64}(lCV),
#     ), 1)

#     HomotopyDensity{
#         typeof(reprkind),
#         P, 
#         P,
#         Matrix{Float64},
#         eltype(trailing_forms),
#     }(;
#         reprkind,
#         points,
#         trailing_forms,
#         kw...
#     )
# end

@deprecate HomotopyDensity_legacy(;
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
  manifold::M, 
  points::Vector{P},
  kernel_bw = nothing,
  kw...
) where {M, P} HomotopyDensity_legacy(manifold, points; partial, kernel_bw, kw...)

@deprecate getPointRepr(x::HomotopyDensity) getPointType(x)

@deprecate getInfoPerCoord(mkd::HomotopyDensity, aspartial::Bool = true) getObservability(mkd, aspartial)

@deprecate exists_BTLabel(hode::HomotopyDensity, idx::Int) isassigned(hode, idx)


export manikde!

function manikde!(
  manifold::AbstractManifold,
  pts::AbstractVector;
  bw = diagm(ones(manifold_dimension(manifold))),
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
  kw...
) 
  _partial = _tuple(partial)
  @warn "manikde!(manif, pts; partial, kw...) is deprecated, use HomotopyDensity_legacy(manif, pts; partial, kw...) instead"
  HomotopyDensity_legacy(
    manifold,
    pts;
    partial = _partial,
    bw,
    kw...
  )
end



# # EXPERIMENTAL, untested, likely buggy
# function childIndices(
#     mt::HomotopyDensity, 
#     krnIdx::Int;
#     mixturedepth::Int = 999,
# )
#     N = Npts(mt)
#     btleft = 2 * krnIdx
#     # FIXME, isleaf for right children only (can happen when doing geometric split)
#     # e.g. for N=length(data)=32, left child of 1*2 = 2, and left child of 2*2=4, whose left child is 4*2 = 8, similarly 8*2=16.  
#     #  Now the left child of node 16*2 = 32, which is the first leaf node (but careful with index == N)
#     #  i.e. right child of node 15 is 2*15+1 = 31, so 15's right child (31) is the last nonleaf
#     isleaf = N <= btleft
#     # Before BeliefTreeIndices nonisleaf are [1..N], while isleaf are [N+1..2N].
#     left = btleft + (isleaf ? 1 : 0) 
#     nonleaf_left = isleaf ? -1 : btleft
#     leaf_left = isleaf ? nonleaf_left : -1
#     right = left + 1 
#     nonleaf_right = isleaf ? -1 : nonleaf_left + 1
#     leaf_right = isleaf ? nonleaf_left + 1 : -1
#     # return a pseudo type representing a composite index of the belief tree
#     left_ci = (;
#         nonleaf_left,
#         leaf_left,
#         isleaf,
#         # TBD permuted indices?
#     )
#     right_ci = (;
#         nonleaf_right,
#         leaf_right,
#         isleaf,
#         # TBD permuted indices?
#     )
#     return (;
#         left_ci,
#         right_ci,
#         # legacy values below
#         N,
#         left,
#         right, 
#     )
# end