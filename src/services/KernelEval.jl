
# also makes static
function projectSymPosDef(c::AbstractMatrix)
  s = size(c)
  # pretty fast to make or remake isbitstype form matrix
  _c = SMatrix{s...}(c)
  #TODO likely not intended project here: see AMP#283
  issymmetric(_c) ? _c : project(SymmetricPositiveDefinite(s[1]),_c,_c)
end

function MvNormalKernel(
  μ::AbstractArray,
  Σ::AbstractArray,
  weight::Real=1.0
)
  _c = projectSymPosDef(Σ)
  p=MvNormal(_c)
  # NOTE, TBD, why not sqrt(inv(p.Σ)), this had an issue seemingly internal to PDMat.chol which breaks an already forced SymPD matrix to again be not SymPD???
  sqrt_iΣ = sqrt(inv(_c)) 
  MvNormalKernel(;μ, p, sqrt_iΣ, weight=float(weight))
end

Statistics.mean(m::MvNormalKernel) = m.μ # mean(m.p) # m.p.μ
Statistics.cov(m::MvNormalKernel) = cov(m.p) # note also about m.sqrt_iΣ
Statistics.std(m::MvNormalKernel) = sqrt(cov(m))

updateKernelBW(k::MvNormalKernel,_bw) = (p=MvNormal(_bw); MvNormalKernel(;μ=k.μ,p,weight=k.weight))

function evaluate(
  M::AbstractManifold,
  ekr::MvNormalKernel,
  p # on manifold point
)
  #
  dim = manifold_dimension(M)
  nscl = 1/sqrt((2*pi)^dim * det(cov(ekr)))
  return nscl * ker(M, ekr, p, 0.5, distanceMalahanobisSq)
end


"""
    $SIGNATURES

Transform `T=RS` from unit covariance `D` to instance covariance `Σ = TD`.

Notes:
- Geometric interpretation of the covariance matrix, Fig. 10, https://users.cs.utah.edu/~tch/CS6640F2020/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf
  - Eigen decomp: `Σ^2 V = VL` => `Σ^2 = VL(V^-1) = RL(R^-1) = RSS(R^-1)` => `T=RS`
"""
function covTransformNormalized(Σ::AbstractMatrix)
  F = eigen(Σ)
  R = F.vectors
  L = diagm(F.values)
  S = sqrt(L)
  return R*S
end

function Base.show(io::IO, mvk::MvNormalKernel)
  μ = mean(mvk)
  Σ2 = cov(mvk)
  # Σ=sqrt(Σ2)
  d = size(Σ2,1)
  print(io, "MvNormalKernel(d=",d)
  print(io,",μ=",round.(μ;digits=3))
  print(io,",Σ^2=[",round(Σ2[1];digits=3))
  if 1<d
    print(io,"...")
  end
  # det(T-I) is a proxy through volume meaure of Transform from unit covariance matrix to this instance
  # i.e. how large or rotated is this covariance instance
  println(io,"]); det(T-I)=",round(det(covTransformNormalized(Σ2)-diagm(ones(d)));digits=3)) 
    # ; det(Σ)=",round(det(Σ);digits=3), "
  nothing
end

Base.show(io::IO, ::MIME"text/plain", mvk::MvNormalKernel) = show(io, mvk)


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

