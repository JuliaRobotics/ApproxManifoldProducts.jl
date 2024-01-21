
# also makes static
function projectSymPosDef(c::AbstractMatrix)
  s = size(c)
  # pretty fast to make or remake isbitstype form matrix
  _c = SMatrix{s...}(c)
  issymmetric(_c) ? _c : project(SymmetricPositiveDefinite(s[1]),_c,_c)
end


abstract type AbstractKernel end

@kwdef struct MvNormalKernel{T,M,iM} <: AbstractKernel
  p::MvNormal{T,M}
  # TDB might already be covered in p.Σ.chol but having issues with SymPD (not particular to this AMP repo)
  sqrt_iΣ::iM = sqrt(inv(p.Σ))
end

Base.eltype(mt::MvNormalKernel) = eltype(mt.p)

function MvNormalKernel(m::AbstractVector,c::AbstractArray)
  _c = projectSymPosDef(c)
  p=MvNormal(m,_c)
  # NOTE, TBD, why not sqrt(inv(p.Σ)), this had an issue seemingly internal to PDMat.chol which breaks an already forced SymPD matrix to again be not SymPD???
  sqrt_iΣ = sqrt(inv(_c)) 
  MvNormalKernel(;p, sqrt_iΣ)
end

Statistics.mean(m::MvNormalKernel) = m.p.μ
Statistics.cov(m::MvNormalKernel) = m.p.Σ # note also about m.sqrt_iΣ


function distanceMalahanobisCoordinates(
  M::AbstractManifold, 
  K::AbstractKernel, 
  q,
  basis=DefaultOrthogonalBasis()
)
  p = mean(K)
  i_p = inv(M,p)
  pq = Manifolds.compose(M, i_p, q)
  ϵ = identity_element(M,q)
  X = log(M, ϵ, pq)
  Xc = get_coordinates(M, ϵ, X, basis)
  return K.sqrt_iΣ*Xc
end

function distanceMalahanobisSq(
  M::AbstractManifold,
  K::AbstractKernel,
  q,
  basis=DefaultOrthogonalBasis()
)
  δc = distanceMalahanobisCoordinates(M,K,q,basis)
  p = mean(K)
  ϵ = identity_element(M, q)
  X = get_vector(M, ϵ, δc, basis)
  return inner(M, p, X, X)
end


function _distance(
  M::AbstractManifold, 
  p::AbstractVector, 
  q::AbstractVector, 
  kernel = (_p) -> MvNormalKernel(
    p=MvNormal(_p,SVector(ntuple((s)->1,manifold_dimension(M))...))
  ),
  distFnc::Function=distanceMalahanobisSq, 
  # distFnc::Function=distanceMalahanobisSq,
)
  distFnc(M, kernel(p), q)
end


"""
$SIGNATURES

Normal kernel used for Hilbert space embeddings.
"""
ker(M::AbstractManifold, p, q, sigma::Real=0.001, distFnc=(_M,_p,_q)->distance(_M,_p,_q)^2) = exp( -sigma*distFnc(M, p, q) ) # _distance(M,p,q) # 

