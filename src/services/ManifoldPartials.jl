
export getManifoldPartial

function _checkManifoldPartialDims(M::AbstractManifold, 
                                    partial::AbstractVector{Int}, 
                                    offset::Base.RefValue{Int}, 
                                    doError::Bool=true)
  #
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  doError && !any(mask) && error("Unknown manifold partial=$(partial .- offset[]) in dimension=$(manifold_dimension(M)) of $M")
  return mask
end

function getManifoldPartial(M::Euclidean{Tuple{N}}, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true) where N
  #
  mask = _checkManifoldPartialDims(M,partial,offset, doError)
  offset[] += manifold_dimension(M)
  Euclidean(sum(mask))
end

function getManifoldPartial(M::Circle, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true)
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  M
end

function getManifoldPartial(M::Rotations{2}, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true)
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  M
end

function getManifoldPartial(M::TranslationGroup{Tuple{N}}, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true) where N
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  TranslationGroup(sum(mask))
end

function getManifoldPartial(M::typeof(SpecialOrthogonal(2)), 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true)
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  return M
end

function getManifoldPartial(M::ProductManifold, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true)
  #
  _ = _checkManifoldPartialDims(M,partial,offset,doError)

  # loop through the ProductManifold components 
  ARR = []
  for m in M.manifolds
    mask = _checkManifoldPartialDims(m,partial,offset, false)
    if any(mask)
      push!(ARR, getManifoldPartial(m, partial, offset, false))
    else
      offset[] += manifold_dimension(m)
    end
  end

  # trivial case, drop the ProductManifold for single element
  if length(ARR) == 1
    return ARR[1]
  elseif 1 < length(ARR)
    return ProductManifold(ARR...)
  end
  error("partial manifold calculations should not reach here")
end

function getManifoldPartial(M::GroupManifold, 
                            partial::AbstractVector{Int}, 
                            offset::Base.RefValue{Int}=Ref(0),
                            doError::Bool=true)
  #
  mask = _checkManifoldPartialDims(M,partial,offset, doError)
  if sum(mask) == manifold_dimension(M)
    # asking for all coordinate dimensions as offered by M
    return M
  end
  # recursion may need to branch for ProductManifold
  # Note loss of the Group operation information at this time
  getManifoldPartial(M.manifold, partial, offset, doError)
end


# # assumed to follow coordinates as transition step towards more general solution
# # assume ProductManifold is the only way to stitch multiple manifolds together
# function getManifoldPartial(M::MB.AbstractManifold, 
#                             partial::AbstractVector{Int}, 
#                             offset::Base.RefValue{Int}=Ref(0)  )
#   #
#   # recursive exit criterion
#   if maximum(partial) < offset[]
#     return nothing
#   end

#   # find which dimensions should be added
#   shifted = partial .- offset
#   mask = shifted .<= manifold_dimension(M)
#   if any(mask)
#     return getManifoldPartial(M, shifted[mask])
#   end

#   @error "unknown manifold partial situation" M partial offset
#   error("unknown manifold partial situation,\n$M\n$partial\n$offset")
# end
