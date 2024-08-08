

abstract type AbstractKernel end

@kwdef struct MvNormalKernel{P,T,M,iM} <: AbstractKernel
  """ On-manifold point representing center (mean) of the MvNormal distrubtion """ 
  μ::P
  """ Zero-mean normal distrubtion with covariance """
  p::MvNormal{T,M}
  # TDB might already be covered in p.Σ.chol but having issues with SymPD (not particular to this AMP repo)
  """ Manually maintained square root concentration matrix for faster compute, TODO likely duplicate of existing Distrubtions.jl functionality. """
  sqrt_iΣ::iM = sqrt(inv(cov(p)))
  """ Nonparametric weight value """
  weight::Float64 = 1.0
end
