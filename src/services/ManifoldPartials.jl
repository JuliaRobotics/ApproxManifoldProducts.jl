
export getManifoldPartial

# forcing ProductManifold to use ProductRepr as accompanying representation
const _PartiableRepresentationProduct = Union{Nothing,<:ProductRepr}
# forcing ProductManifold to use ProductRepr as accompanying representation
const _PartiableRepresentationFlat{T} = Union{Nothing,<:AbstractVector{T}}
# More general representation for Manifold Factors or Groups
const _PartiableRepresentation = Union{<:_PartiableRepresentationProduct,<:_PartiableRepresentationFlat,<:AbstractMatrix}

## COMMON UTILS FOR PARTIAL MANIFOLDS

function _checkManifoldPartialDims( M::AbstractManifold, 
                                    partial::AbstractVector{Int}, 
                                    offset::Base.RefValue{Int},
                                    doError::Bool=true)
  #
  mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
  doError && !any(mask) && error("Unknown manifold partial=$(partial .- offset[]) in dimension=$(manifold_dimension(M)) of $M")
  return mask
end

## EXTRACT REPRESENTATION PARTIALS

# do nothing case
_getReprPartial(M::MB.AbstractManifold, ::Nothing, w...; kw...) = nothing

# drop back to regular vector
function _getReprPartial( M::MB.AbstractManifold, 
                          repr::AbstractVector{T}, 
                          partial::AbstractVector{Int}, # total partial from user over all Factors
                          offset::Base.RefValue{Int}=Ref(0),
                          mask::BitVector=_checkManifoldPartialDims(M,partial,offset, doError) ) where {T <: Number}
  #
  ret = zeros(T, sum(mask))
  for (i,p) in enumerate(partial .- offset)
    ret[i] = repr[p]
  end
  return ret
end

function _getReprPartial( M::Union{<:typeof(SpecialOrthogonal(2)), <:Rotations{2}}, 
                          repr::AbstractMatrix{T}, 
                          partial::AbstractVector{Int}, # total partial from user over all Factors
                          offset::Base.RefValue{Int}=Ref(0),
                          mask::BitVector=_checkManifoldPartialDims(M,partial,offset, false) ) where {T <: Number}
  #
  @assert sum(mask) == 1 "Can only return the same point represenation matrix for SpecialOrthogonal(2) / Rotations(2)"
  return repr
end


## EXTRACT PARTIAL MANIFOLD


function getManifoldPartial(M::Euclidean{Tuple{N}}, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentationFlat{T}=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true) where {N, T <: Number}
  #
  mask = _checkManifoldPartialDims(M,partial,offset, doError)
  offset[] += manifold_dimension(M)
  len = sum(mask)
  repr_p = repr === nothing ? nothing : zeros(T,len) 
  return (Euclidean(len),repr_p)
end

function getManifoldPartial(M::Circle, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentation=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true )
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  return (M,repr)
end

function getManifoldPartial(M::Rotations{2}, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentation=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true )
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  return (M,repr)
end

function getManifoldPartial(M::TranslationGroup{Tuple{N}}, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentationFlat{T}=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true ) where {N, T <: Number}
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  len = sum(mask)
  repr_p = repr === nothing ? nothing : zeros(T,len)
  return (TranslationGroup(len),repr_p)
end

function getManifoldPartial(M::typeof(SpecialOrthogonal(2)), 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentation=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true )
  #
  mask = _checkManifoldPartialDims(M,partial,offset,doError)
  offset[] += manifold_dimension(M)
  return (M,repr)
end

# assumed to follow coordinates as transition step towards more general solution
# assume ProductManifold is the only way to stitch multiple manifolds together
function getManifoldPartial(M::ProductManifold, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentationProduct=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true )
  #
  _checkManifoldPartialDims(M,partial,offset,doError)

  # loop through the ProductManifold components 
  ManiArr = []
  ReprArr = []
  for (i,m) in enumerate(M.manifolds)
    mask = _checkManifoldPartialDims(m,partial,offset, false)
    if any(mask)
      Mp = if repr === nothing
        # decide if representation should also be updated or left as nothing
        Mp, = getManifoldPartial(m, partial, nothing, offset, doError=false)
        Mp
      else
        # hard assumption that repr::ProductRepr to go along with M::ProductManifold
        Mp, Rp = getManifoldPartial(m, partial, repr.parts[i], offset, doError=false)
        push!(ReprArr, Rp)
        Mp
      end
      push!(ManiArr, Mp)
    else
      offset[] += manifold_dimension(m)
    end
  end

  # trivial case, drop the ProductManifold for single element
  if length(ManiArr) == 1
    repr_p = repr === nothing ? nothing : ReprArr[1]
    return (ManiArr[1],repr_p)
  elseif 1 < length(ManiArr)
    repr_p = repr === nothing ? nothing : ProductRepr(ReprArr...)
    return (ProductManifold(ManiArr...), repr_p)
  end
  error("partial manifold calculations should not reach here")
end

function getManifoldPartial(M::GroupManifold, 
                            partial::AbstractVector{Int}, 
                            repr::_PartiableRepresentation=nothing,
                            offset::Base.RefValue{Int}=Ref(0);
                            doError::Bool=true )
  #
  # mask the desired coordinate dimensions
  mask = _checkManifoldPartialDims(M,partial,offset, doError)

  if sum(mask) == manifold_dimension(M)
    # asking for all coordinate dimensions as offered by M
    return (M,repr)
  end
  # recursion may need to branch for ProductManifold
  # Note loss of the Group operation information at this time
  getManifoldPartial(M.manifold, partial, repr, offset, doError=doError)
end



#