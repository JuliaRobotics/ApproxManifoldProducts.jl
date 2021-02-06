# Interface

mutable struct ManifoldKernelDensity{M <: MB.Manifold{MB.ℝ}, B}
  manifold::Type{M}
  belief::B
end
ManifoldKernelDensity(m::M,b::B) where {M <: MB.Manifold{MB.ℝ}, B} = ManifoldKernelDensity{M,B}

const MKD{M,B} = ManifoldKernelDensity{M, B}

@deprecate ManifoldBelief(w...;kw...) ManifoldKernelDensity(w...;kw...)



function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.Manifold{MB.ℝ},B}
  @info "taking manifold product of $(length(PP)) terms, $M, $B"
  @error "No known product definition"
end
