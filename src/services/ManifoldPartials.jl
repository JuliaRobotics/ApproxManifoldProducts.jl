
export getManifoldPartial


function getManifoldPartial(M::Euclidean{Tuple{N}}, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0)) where N
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  offset[] += manifold_dimension(M)
  Euclidean(sum(mask))
end

function getManifoldPartial(M::Circle, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0))
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  # @assert (offset[]+1) in partial "Circle() can only take partial=[1;], not $partial"
  offset[] += manifold_dimension(M)
  M
end

function getManifoldPartial(M::Rotations{2}, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0))
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  # @assert (offset[]+1) in partial "Rotations(2) can only take partial=[1;], not $partial"
  offset[] += manifold_dimension(M)
  M
end

function getManifoldPartial(M::TranslationGroup{Tuple{N}}, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0)) where N
  # @assert maximum(partial .- offset) <= N "TranslationGroup($N) cannot take partial of larger dimension $partial"
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  offset[] += manifold_dimension(M)
  TranslationGroup(sum(mask))
end

function getManifoldPartial(M::typeof(SpecialOrthogonal(2)), partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0))
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  # if sum(mask) == 0
  #   return nothing
  # end
  offset[] += manifold_dimension(M)
  return M
end

function getManifoldPartial(M::ProductManifold, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0))
  
  # loop through the ProductManifold components 
  ARR = []
  for m in M.manifolds
    # @info "ProductManifold" m manifold_dimension(m) offset[] string(partial)
    mask = 0 .< (partial .- offset[]) .<= manifold_dimension(m)
    if any(mask)
      push!(ARR, getManifoldPartial(m, partial, offset))
    else
      offset[] += manifold_dimension(m)
    end
  end

  # trivial case, drop the ProductManifold for single element
  if length(ARR) == 1
    return ARR[1]
  elseif 1 < length(ARR)
    return ProductManifold(ARR...)
  else
    return nothing
  end
end


# i.e. has field .manifold
function getManifoldPartial(M::GroupManifold, partial::AbstractVector{Int}, offset::Base.RefValue{Int}=Ref(0))
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  if sum(mask) == manifold_dimension(M)
    # asking for all coordinate dimensions as offered by M
    return M
  end
  # recursion may need to branch for ProductManifold
  # hasfield .manifold
  # Note loss of the Group operation information at this time
  getManifoldPartial(M.manifold, partial, offset)
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
