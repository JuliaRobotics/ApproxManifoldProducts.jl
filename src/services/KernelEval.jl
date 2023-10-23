
abstract type AbstractKernel end

@kwdef struct MvNormalKernel{T,M} <: AbstractKernel
  p::MvNormal{T,M}
  sqrt_iΣ::M = sqrt(inv(p.Σ))
end


Statistics.mean(m::MvNormalKernel) = m.p.μ

function distanceMalahanobisCoordinates(
  M::AbstractManifold, 
  K::AbstractKernel, 
  q,
  basis=DefaultOrthonormalBasis()
)
  p = mean(K)
  i_p = inv(M,p)
  pq = Manifolds.compose(M, i_p, q)
  X = log(M, p, pq)
  Xc = get_coordinates(M, p, X, basis)
  return K.sqrt_iΣ*Xc
end

function distanceMalahanobisSq(
  M::AbstractManifold,
  K::AbstractKernel,
  q,
  basis=DefaultOrthonormalBasis()
)
  δc = distanceMalahanobisCoordinates(M,K,q,basis)
  p = mean(K)
  # ϵ = identity_element(M, q)
  X = get_vector(M, p, δc, basis)
  return inner(M, p, X, X)
end

# function distance(
#   M::AbstractManifold, 
#   p::AbstractVector, 
#   q::AbstractVector, 
#   kernel=(_p) -> MvNormalKernel(
#     p=MvNormal(_p,SVector(ones(manifold_dimension(M))...))
#   ), 
#   distFnc::Function=distanceMalahanobisSq
# )
#   distFnc(M, kernel(p), q)
# end

"""
    $SIGNATURES

Normal kernel used for Hilbert space embeddings.
"""
ker(M::AbstractManifold, p, q, sigma::Real=0.001) = exp( -sigma*(distance(M, p, q)) )

