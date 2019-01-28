# Interface


abstract type Manifold end


mutable struct ManifoldBelief{M <: Manifold, B}
  manifold::Type{M}
  belief::B
end
ManifoldBelief(m::M,b::B) where {M <: Manifold, B} = ManifoldBelief{M,B}

const MB{M,B} = ManifoldBelief{M, B}

function *(PP::Vector{MB{M,B}}) where {M<:Manifold,B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end
