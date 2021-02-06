# Interface

mutable struct ManifoldBelief{M <: MB.Manifold{MB.ℝ}, B}
  manifold::Type{M}
  belief::B
end
ManifoldBelief(m::M,b::B) where {M <: MB.Manifold{MB.ℝ}, B} = ManifoldBelief{M,B}

const MB_{M,B} = ManifoldBelief{M, B}

function *(PP::Vector{MB_{M,B}}) where {M<:MB.Manifold{MB.ℝ},B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end
