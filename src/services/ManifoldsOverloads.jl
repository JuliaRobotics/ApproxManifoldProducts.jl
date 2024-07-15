
const _UPSTREAM_MANIFOLDS_ADJOINT_ACTION = false

# local union definition during development -- TODO consolidate upstream
LieGroupManifoldsPirate = Union{typeof(SpecialOrthogonal(2)), typeof(SpecialOrthogonal(3)), typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}

## ================================ GENERIC IMPLEMENTATIONS ================================


function TUs.skew(v::SVector{3,T}) where T<:Real
  x,y,z = v[1], v[2], v[3]
  return SMatrix{3,3,T}(
    +0,  
    z, 
    -y,
    -z,  
    0,  
    x,
    +y, 
    -x,  
    0
  )
end


# right Jacobian (Lie Group, originally from ?)
function Jr(M::Manifolds.GroupManifold, X; order=5)
  adx = ad(M, X)
  mapreduce(+, 0:order) do i
      (-adx)^i / factorial(i + 1)
  end
end

# "The left trivialized Jacobian is referred to as the right Jacobian in much of the key literature." [Ge, van Goor, Mahony, 2024]
# argument d is a Lie algebra element giving the direction of transport 
function Jl_trivialized(
  M::Manifolds.GroupManifold, 
  d
)
  adM = ad(M,d)
  ead = exp(-adM)
  adM \ (LinearAlgebra.I - ead)
end


"""
    $SIGNATURES

EXPERIMENTAL: ApproxManifoldProducts hard coded versions for best parallel transport available (as per development cycle).

Inputs:
- M is a Manifold (must be a Lie group)
- p is the expansion point on manifold
- X is a tangent vector which is to be transported
- d is a tangent vector from a starting point in the direction and distance to transport 

Sidenote on why lowercase tangent vector `d` (as driven by Manifolds.jl)
```
so(2)
X = Xc*e1 = Xc*î
î = ∂/∂x = [0 -1; 0 1] # orthogonal basis
```
Notes
- Default is transport without curvature estimate provided by upstream Manifolds.jl 

See also: [`Jl_trivialized'](@ref), [`Jr`](@ref), `Manifolds.parallel_transport_direction`, `Manifolds.parallel_transport_to`, `Manifolds.parallel_transport_along`
"""
parallel_transport_best(
  M::AbstractManifold,
  p,
  X::AbstractArray,
  d::AbstractArray
) = Manifolds.parallel_transport_direction(M, p, X, d)



## ================================= Lie (a/A)djoints =================================
## ---------------------------------- Almost generic ----------------------------------

# assumes inputs are Lie algebra tangent vectors represented in matrix form
ad(
  M::LieGroupManifoldsPirate, 
  A::AbstractMatrix, 
  B::AbstractMatrix
) = Manifolds.lie_bracket(M, A, B)


Ad(
  M::Union{typeof(SpecialOrthogonal(2)), typeof(SpecialOrthogonal(3))}, 
  p, 
  X::AbstractMatrix;
  use_upstream::Bool = _UPSTREAM_MANIFOLDS_ADJOINT_ACTION # explict to support R&D
) = if use_upstream
  Manifolds.adjoint_action(M, p, X)
else
  p*X*(p')
end


function Ad(
  M::Union{typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}, 
  p, 
  X::ArrayPartition;
  use_upstream::Bool = _UPSTREAM_MANIFOLDS_ADJOINT_ACTION # explict to support R&D
)
  if use_upstream
    # TODO swap and test
    Manifolds.adjoint_action(M, p, X)
  else
    t = p.x[1]
    R = p.x[2]
    v = X.x[1]
    Ω = X.x[2]
    # direct adjoint for SO(.)
    RΩR = Ad(M.manifold[2], R, Ω; use_upstream )
    ArrayPartition(-RΩR*t + R*v, RΩR)
  end
end


# left trivialized Jacobian [Mahony, 2024, Theorem 4.3, matrix P_u_vee]
# U is direction in which to transport along, given as Lie algebra element (ie tangent vector) 
# this produces the transportMatrix P_u^vee, per [Mahony, 2024]
# d is a Lie algebra element giving the direction of transport 
parallel_transport_direction_lie(
  M::typeof(SpecialOrthogonal(3)),
  d::AbstractMatrix
) = exp(ad(M, -0.5*d))


# transport with 2nd order curvature approximation
# left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
parallel_transport_along_2nd(
  M::LieGroupManifoldsPirate,
  p,
  X::AbstractMatrix,
  d::AbstractMatrix
) = Jl_trivialized(M, d) * vee(M,p,X)


parallel_transport_best(
  M::LieGroupManifoldsPirate,
  p,
  X::AbstractMatrix,
  d::AbstractMatrix
) = parallel_transport_along_2nd(M,p,X,d)

## ----------------------------- SpecialOrthogonal(3) -----------------------------


# matrix versions
# [Chirikjian, 2012 Vol2, p.39]
ad(
  ::typeof(SpecialOrthogonal(3)), 
  X
) = X

Ad(
  ::typeof(SpecialOrthogonal(3)), 
  R
) = R


## For SO(3) Jr and Jl + inv closed forms, see [Chirikjian 2012, Vol2, Vol2 p.40] !!!


# Matrix form parallel transport with curvature correction (2nd order) 
# left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
# U is direction in which to transport (w/ 2nd order along curvature correction), 
# Direction d is a Lie algebra element (tangent vector) providing direction of transport 
function Jl_trivialized(
  M::typeof(SpecialOrthogonal(3)), 
  d
)
  aduv = ad(M, -0.5*d)
  P = exp(aduv)
  return P*(LinearAlgebra.I + 1/24*aduv^2)
end

# d: Lie algebra for the direction of transport
# Xc are coordinates to be transported
Jl_trivialized(
  M::typeof(SpecialOrthogonal(3)), 
  d::AbstractMatrix, 
  Xc::AbstractVector
) = Jl_trivialized(M, d)*Xc



function parallel_transport_direction_lie(
  M::typeof(SpecialOrthogonal(3)), 
  d::AbstractMatrix, 
  X
)
  hat(
    M, 
    Identity(M), 
    parallel_transport_direction_lie(M, d) * vee(M, Identity(M), X)
  )
end



## ----------------------------- SpecialEuclidean(.) -----------------------------


# assumes inputs are Lie algebra tangent vectors represented in (Array) Partion form
# d is a Lie algebra element (tangent vector) providing the direction of transport
# X is the tangent vector to be transported 
function ad(
  M::Union{typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}, 
  d::ArrayPartition, 
  X::ArrayPartition
)
  #
  v1x = skew(d.x[1])
  Ω1  = d.x[2]
  v2 = X.x[1]
  ω2 = log_lie(M.manifold[2], X.x[2])

  Rε = identity_element(M.manifold[2], Ω1)
  Ω = Manifolds.hat(M.manifold[2], Rε, Ω1*ω2)

  ArrayPartition(v1x*ω2 + Ω1*v2, Ω)
end


# matrix versions


function ad(
  ::typeof(SpecialEuclidean(2)), 
  d::ArrayPartition, 
)
  Vx = SA[d.x[2]; -d.x[1]]
  Ω = d.x[2]
  vcat(
    hcat(Ω, Vx),
    zero(SMatrix{1,3,Float64})
  )
end

function ad(
  ::typeof(SpecialEuclidean(3)), 
  d::ArrayPartition, 
)
  Vx = skew(d.x[1])
  Ω = d.x[2]
  vcat(
    hcat(Ω, Vx),
    hcat(zero(SMatrix{3,3,Float64}), Ω)
  )
end


function Ad(
  ::typeof(SpecialEuclidean(2)), 
  p
)
  t = p.x[1]
  R = p.x[2]
  vcat(
      hcat(R, -SA[0 -1; 1 0]*t),
      SA[0 0 1]
  )
end

function Ad(
  ::typeof(SpecialEuclidean(3)), 
  p
)
  t = p.x[1]
  R = p.x[2]
  vcat(
      hcat(R, skew(t)*R),
      hcat(zero(SMatrix{3,3,Float64}), R)
  )
end




#