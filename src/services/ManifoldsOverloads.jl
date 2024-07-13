
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


# Lie algebra adjoints

LieGroupManifolds = Union{typeof(SpecialOrthogonal(2)), typeof(SpecialOrthogonal(3)), typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}

# assumes inputs are Lie algebra tangent vectors represented in matrix form
adjoint(
  M::LieGroupManifolds, 
  A::AbstractMatrix, 
  B::AbstractMatrix
) = Manifolds.lie_bracket(M, A, B)

# assumes inputs are Lie algebra tangent vectors represented in (Array) Partion form
function adjoint(
  M::Union{typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}, 
  S1::ArrayPartition, 
  S2::ArrayPartition
)
  #
  v1x = skew(S1.x[1])
  Ω1  = S1.x[2]
  v2 = S2.x[1]
  ω2 = log_lie(M.manifolds[2], S2.x[2])

  Rε = identity_element(M.manifolds[2], Ω1)
  Ω = Manifolds.hat(M.manifolds[2], Rε, Ω1*ω2)

  ArrayPartition(v1x*ω2 + Ω1*v2, Ω)
end



# Lie Group Adjoints
# [Chirikjian, 2012 Vol2, p.39]
adjointMatrix(::typeof(SpecialOrthogonal(3)), X) = X
AdjointMatrix(::typeof(SpecialOrthogonal(3)), R) = R


Ad(M::Union{typeof(SpecialOrthogonal(2)), typeof(SpecialOrthogonal(3))}, R, Ω) = R*Ω*R'

function Ad(M::Union{typeof(SpecialEuclidean(2)), typeof(SpecialEuclidean(3))}, p, X)
  if true
    t = p.x[1]
    R = p.x[2]
    v = X.x[1]
    Ω = X.x[2]
    # direct adjoint for SO(.)
    RΩR = Ad(M.manifold[2], R, Ω )
    ArrayPartition(-RΩR*t + R*v, RΩR)
  else
    # TODO swap and test
    Manifolds.adjoint_action(M, p, X)
  end
end

# Ad_vee?
function Ad(::typeof(SpecialEuclidean(3)), p)
  t = p.x[1]
  R = p.x[2]
  vcat(
      hcat(R, skew(t)*R),
      hcat(zero(SMatrix{3,3,Float64}), R)
  )
end

function Ad(::typeof(SpecialEuclidean(2)), p)
  t = p.x[1]
  R = p.x[2]
  vcat(
      hcat(R, -SA[0 -1; 1 0]*t),
      SA[0 0 1]
  )
end



# right Jacobian (Lie Group, originally from ?)
function Jr(M::Manifolds.GroupManifold, X; order=5)
  adx = ad(M, X)
  mapreduce(+, 0:order) do i
      (-adx)^i / factorial(i + 1)
  end
end


# # from [Ge, van Goor, Mahony, 2024]
# function approxJacobianLieGroup()
#   Ad_vee()
# end


# left trivialized Jacobian [Mahony, 2024, Theorem 4.3, matrix P_u_vee]
# U is direction in which to transport along, given as Lie algebra element (ie tangent vector) 
transportMatrix(M::typeof(SpecialOrthogonal(3)),U::AbstractMatrix) = exp(adjointMatrix(M, -0.5*U))

function transportAction_matrices_direction(M::typeof(SpecialOrthogonal(3)), U::AbstractMatrix, W)
  Yc = transportMatrix(M, U) * vee(M, Identity(M), W)
  hat(M, Identity(M), Yc)
end


# Matrix form parallel transport with curvature correction (2nd order) 
# left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
# U is direction in which to transport (w/ 2nd order along curvature correction), 
# Direction U is given as Lie algebra element (ie tangent vector) 
function Jl_trivialized(M::typeof(SpecialOrthogonal(3)), U)
  aduv = adjointMatrix(M, -0.5*U)
  P = exp(aduv)
  return P*(LinearAlgebra.I + 1/24*aduv^2)
end

# U is the direction
Jl_trivialized(M::typeof(SpecialOrthogonal(3)), U::AbstractMatrix, w::AbstractVector) = Jl_trivialized(M, U)*w

# # argument U is Lie algebra element
# # "The left trivialized Jacobian is referred to as the right Jacobian in much of the key literature." [Ge, van Goor, Mahony, 2024]
# function Jl_trivialized(M::Manifolds.GroupManifold, U)
#   @error "FIXME AMP.Jl_trivialized"
#   advee = ad(M,U)
#   ε = identity_element(M)
#   ead = exp(-advee)
#   I0 = diagm(ones(manifold_dimension(M)))
#   advee \ (I0 - ead)
# end


## For SO(3) Jr and Jl + inv closed forms, see [Chirikjian 2012, Vol2, Vol2 p.40] !!!




# Dehann sidenote
# so(2)
# X = Xc*e1 = Xc*î
# î = ∂/∂x = [0 -1; 0 1] # orthogonal basis
