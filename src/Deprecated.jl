

## ======================================================================================================
## Remove below before v0.16
## ======================================================================================================

export manikde!

function manikde!(
  manifold::AbstractManifold,
  pts::AbstractVector;
  bw = diagm(ones(manifold_dimension(manifold))),
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
  kw...
) 
  _partial = _tuple(partial)
  @warn "manikde!(manif, pts; partial, kw...) is deprecated, use HomotopyDensity{partial}(manif, pts; kw...) instead"
  HomotopyDensity{
    _partial
  }(
    manifold,
    pts;
    bw,
    kw...
  )
end