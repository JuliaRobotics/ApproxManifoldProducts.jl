


## ======================================================================================================
## Remove below before v0.17
## ======================================================================================================

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