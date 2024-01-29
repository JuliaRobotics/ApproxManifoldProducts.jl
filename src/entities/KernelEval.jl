

abstract type AbstractKernel end

@kwdef struct MvNormalKernel{T,M,iM} <: AbstractKernel
  p::MvNormal{T,M}
  # TDB might already be covered in p.Σ.chol but having issues with SymPD (not particular to this AMP repo)
  sqrt_iΣ::iM = sqrt(inv(p.Σ))
end
