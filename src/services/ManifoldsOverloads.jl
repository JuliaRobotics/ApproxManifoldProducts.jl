
const _UPSTREAM_MANIFOLDS_ADJOINT_ACTION = false

# local union definition during development -- TODO consolidate upstream
LieGroupManifoldsPirate = Union{
    typeof(TranslationGroup(1)),
    typeof(TranslationGroup(2)),
    typeof(TranslationGroup(3)),
    typeof(TranslationGroup(4)),
    typeof(TranslationGroup(5)),
    typeof(TranslationGroup(6)),
    typeof(SpecialOrthogonalGroup(2)),
    typeof(SpecialOrthogonalGroup(3)),
    typeof(SpecialEuclideanGroup(2; variant = :right)),
    typeof(SpecialEuclideanGroup(3; variant = :right)),
}

## ===================================== BASIS PIRATES =====================================

# Sidenote on why lowercase tangent vector `d` (as driven by Manifolds.jl)
# ```
# so(2)
# X = Xc*e1 = Xc*î
# î = ∂/∂α = [0 -1; 0 1] # orthogonal basis
# ```

function get_basis_affine(::TranslationGroup{ℝ, TypeParameter{Tuple{N}}}) where {N}
    return map(i -> SVector{N, Float64}(ntuple(s -> float(s == i), N)), 1:N)
end

get_basis_affine(::typeof(SpecialOrthogonalGroup(2))) = tuple(SA[0 -1; 1 0.0])

function get_basis_affine(::typeof(SpecialOrthogonalGroup(3)))
    return tuple(
        SA[0 -0 0; 0 0 -1; -0 1 0.0],
        SA[0 -0 1; 0 0 -0; -1 0 0.0],
        SA[0 -1 0; 1 0 -0; -0 0 0.0],
    )
end

# right variant is translate-then-rotate
function get_basis_affine(::typeof(SpecialEuclideanGroup(2; variant = :right)))
    return tuple(
        SA[0 -0 1; 0 0 0; 0 0 0.0],
        SA[0 -0 0; 0 0 1; 0 0 0.0],
        SA[0 -1 0; 1 0 0; 0 0 0.0],
    )
end

# right variant is translate-then-rotate
function get_basis_affine(::typeof(SpecialEuclideanGroup(3; variant = :right)))
    return tuple(
        SA[0 -0 0 1; 0 0 -0 0; -0 0 0 0; 0 0 0 0.0],
        SA[0 -0 0 0; 0 0 -0 1; -0 0 0 0; 0 0 0 0.0],
        SA[0 -0 0 0; 0 0 -0 0; -0 0 0 1; 0 0 0 0.0],
        SA[0 -0 0 0; 0 0 -1 0; -0 1 0 0; 0 0 0 0.0],
        SA[0 -0 1 0; 0 0 -0 0; -1 0 0 0; 0 0 0 0.0],
        SA[0 -1 0 0; 1 0 -0 0; -0 0 0 0; 0 0 0 0.0],
    )
end

## ================================ GENERIC IMPLEMENTATIONS ================================


# right Jacobian (Lie Group, originally from ?)
function Jr(M::AbstractLieGroup, X; order = 5)
    adx = ad(M, X)
    return mapreduce(+, 0:order) do i
        (-adx)^i / factorial(i + 1)
    end
end

# # "The left trivialized Jacobian is referred to as the right Jacobian in much of the key literature." [Ge, van Goor, Mahony, 2024]
# # argument d is a Lie algebra element giving the direction of transport 
# function Jr_alt(
#   M::Manifolds.GroupManifold, 
#   d
# )
#   adM = ad(M,d)
#   ead = exp(-adM)
#   TODO confirm left or right inverse here? 
#   adM \ (LinearAlgebra.I - ead)
# end

"""
    $SIGNATURES

EXPERIMENTAL: ApproxManifoldProducts hard coded versions for best parallel transport available (as per development cycle).

Inputs:
- M is a Manifold (must be a Lie group)
- p is the expansion point on manifold
- X is a tangent vector which is to be transported
- d is a tangent vector from a starting point in the direction and distance to transport

Notes
- Default is transport without curvature estimate provided by upstream Manifolds.jl 

Useful references:
- https://www.youtube.com/watch?v=MRU2D6sLpU0

See also: [`parallel_transport_curvature_2nd_lie'](@ref), [`Jr`](@ref), `Manifolds.parallel_transport_direction`, `Manifolds.parallel_transport_to`, `Manifolds.parallel_transport_along`
"""
parallel_transport_best(M::AbstractManifold, p, X::AbstractArray, d::AbstractArray) =
    Manifolds.parallel_transport_direction(M, p, X, d)

## ================================= Lie (a/A)djoints =================================
## ---------------------------------- Almost generic ----------------------------------

_makeaffine(::AbstractManifold, X) = X
function _makeaffine(::SpecialEuclideanGroup, X::ArrayPartition)
    return convert(AbstractMatrix, SpecialEuclideanProductTangentVector(X))
end

# assumes inputs are Lie algebra tangent vectors represented in matrix form
function ad(M::LieGroupManifoldsPirate, X::AbstractMatrix, d::AbstractMatrix)
    return LieGroups.lie_bracket(M, X, d)
end

"""
    $SIGNATURES

Construct generic adjoint matrix for Lie group manifolds.
Input `X` is a tangent vector of the Lie algebra of group manifold `M`.

Notes
- This implementation uses the Lie bracket over affine (or screw) matrices.
  - Ref [Chirikjian 2012, Vol.2, pg.30, eq.10.59b]
- Two parameters means this function builds a matrix that can be used to do the action.
"""
function ad_lie(M::AbstractLieGroup, X::AbstractArray)
    #
    Es = get_basis_affine(M)
    Xa = _makeaffine(M, X)
    𝔤 = LieAlgebra(M)
    return hcat(map(
        (e) -> vee(𝔤, LieGroups.lie_bracket(𝔤, Xa, e)), # Lie bracket here is also the adjoint action for M
        Es,
    )...)
end

# basic fallback
# X is tangent vector (Lie algebra element)
ad(M::LieGroupManifoldsPirate, X) = ad_lie(M, X)

function Ad(
    M::Union{typeof(SpecialOrthogonalGroup(2)), typeof(SpecialOrthogonalGroup(3))},
    p,
    X::AbstractMatrix;
    use_upstream::Bool = _UPSTREAM_MANIFOLDS_ADJOINT_ACTION, # explict to support R&D
)
    if use_upstream
        LieGroups.adjoint_action(M, p, X)
    else
        p * X * (p')
    end
end

function Ad(
    M::Union{
        typeof(SpecialEuclideanGroup(2; variant = :right)),
        typeof(SpecialEuclideanGroup(3; variant = :right)),
    },
    p,
    X::ArrayPartition;
    use_upstream::Bool = _UPSTREAM_MANIFOLDS_ADJOINT_ACTION, # explict to support R&D
)
    if use_upstream
        # TODO swap and test
        LieGroups.adjoint_action(M, p, X)
    else
        t = p.x[1]
        R = p.x[2]
        v = X.x[1]
        Ω = X.x[2]
        # direct adjoint for SO(.)
        RΩR = Ad(M.manifold[2], R, Ω; use_upstream)
        ArrayPartition(-RΩR * t + R * v, RΩR)
    end
end

# left trivialized Jacobian [Mahony, 2024, Theorem 4.3, matrix P_u_vee]
# U is direction in which to transport along, given as Lie algebra element (ie tangent vector) 
# this produces the transportMatrix P_u^vee, per [Mahony, 2024]
# d is a Lie algebra element giving the direction of transport 
function parallel_transport_direction_lie(M::LieGroupManifoldsPirate, d::AbstractMatrix)
    return exp(ad(M, -0.5 * d))
end

# Matrix form parallel transport with curvature correction (2nd order) 
# Jl_trv: the left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
# Direction d is a Lie algebra element (tangent vector) providing direction of transport 
function parallel_transport_curvature_2nd_lie(M::LieGroupManifoldsPirate, d)
    # Lie algebra adjoint matrix
    adx = ad(M, -0.5 * d)
    # parallel_transport_direction_lie (without curvature)
    P = exp(adx) # TBD, is this related to using -0.5d with [LG.diff_left_compose](https://juliamanifolds.github.io/LieGroups.jl/stable/interface/group/#LieGroups.diff_left_compose-Tuple{AbstractLieGroup,%20Any,%20Any,%20Any})?
    # include 2nd order curvature correction
    return P * (LinearAlgebra.I + 1 / 6 * adx^2)
end

# d: Lie algebra for the direction of transport
# Xc are coordinates to be transported
function parallel_transport_curvature_2nd_lie(
    M::LieGroupManifoldsPirate,
    d::AbstractMatrix,
    Xc::AbstractVector,
)
    return parallel_transport_curvature_2nd_lie(M, d) * Xc
end

function parallel_transport_direction_lie(M::LieGroupManifoldsPirate, d::AbstractMatrix, X)
    return hat(
        LieAlgebra(M),
        parallel_transport_direction_lie(M, d) * vee(LieAlgebra(M), X),
    )
end

# transport with 2nd order curvature approximation
# left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
function parallel_transport_along_2nd(
    M::LieGroupManifoldsPirate,
    p,
    X::AbstractMatrix,
    d::AbstractMatrix,
)
    return parallel_transport_curvature_2nd_lie(M, d) * vee(M, p, X)
end

function # TODO default to Manifolds.parallel_transport_along, WIP
parallel_transport_best(M::LieGroupManifoldsPirate, p, X::AbstractMatrix, d::AbstractMatrix)
    return parallel_transport_along_2nd(M, p, X, d)
end # TODO default to Manifolds.parallel_transport_along, WIP

## ----------------------------- SpecialOrthogonalGroup(3) -----------------------------

# matrix versions
# [Chirikjian, 2012 Vol2, p.39]
ad(::typeof(SpecialOrthogonalGroup(3)), X) = X

Ad(::typeof(SpecialOrthogonalGroup(3)), R) = R

## For SO(3) Jr and Jl + inv closed forms, see [Chirikjian 2012, Vol2, Vol2 p.40] !!!

## ----------------------------- SpecialEuclidean(.) -----------------------------

# assumes inputs are Lie algebra tangent vectors represented in (Array) Partion form
# d is a Lie algebra element (tangent vector) providing the direction of transport
# X is the tangent vector to be transported 
function ad(
    M::typeof(SpecialEuclideanGroup(3; variant = :right)),
    d::ArrayPartition,
    X::ArrayPartition,
)
    SO3 = SpecialOrthogonalGroup(3) # TODO use submanifold_component(M, 2) instead?
    v1x = LieGroups.hat(LieAlgebra(SO3), d.x[1]) # skew(d.x[1])
    Ω1 = d.x[2]
    v2 = X.x[1]
    ω2 = log(SO3, X.x[2])

    Ω = hat(LieGroup(SO3), Ω1 * ω2)

    return ArrayPartition(v1x * ω2 + Ω1 * v2, Ω)
end

# matrix versions

function ad(::typeof(SpecialEuclideanGroup(2; variant = :right)), d::ArrayPartition)
    Vx = SA[d.x[1][2]; -d.x[1][1]]
    Ω = d.x[2]
    return vcat(hcat(Ω, Vx), zero(SMatrix{1, 3, Float64}))
end

function ad(::typeof(SpecialEuclideanGroup(3; variant = :right)), d::ArrayPartition)
    v1x = LieGroups.hat(LieAlgebra(SpecialOrthogonalGroup(3)), d.x[1]) # skew(d.x[1])
    Ω = d.x[2]
    return vcat(hcat(Ω, v1x), hcat(zero(SMatrix{3, 3, Float64}), Ω))
end

function Ad(::typeof(SpecialEuclideanGroup(2; variant = :right)), p)
    t = p.x[1]
    R = p.x[2]
    return vcat(hcat(R, -SA[0 -1; 1 0] * t), SA[0 0 1])
end

function Ad(::typeof(SpecialEuclideanGroup(3; variant = :right)), p)
    t = p.x[1]
    R = p.x[2]
    st = LieGroups.hat(LieAlgebra(SO3), t) # skew(t)
    return vcat(hcat(R, st * R), hcat(zero(SMatrix{3, 3, Float64}), R))
end

#
