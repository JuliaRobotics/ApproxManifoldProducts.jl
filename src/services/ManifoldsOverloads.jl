
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

## ================================ GENERIC IMPLEMENTATIONS ================================

# high order truncated series of the left-trivialized Jacobian of the exp map
# NOTE factorial(21) overflows Int64, so order must stay <= 19
function jacobian_exp_series(G::AbstractLieGroup, p, Xp, b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis(); order = 19)
    adx = adjoint_algebra_matrix(G, Xp)
    return mapreduce(+, 0:order) do i
        (-adx)^i / factorial(i + 1)
    end
end

function jacobian_exp_pade_22(G::AbstractLieGroup, p, X, b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis())
    # Small adjoint matrix at the vector X - adₓ
    ad_X = adjoint_algebra_matrix(G, X)
    ad_X2 = ad_X^2

    # [2,2] Padé coefficients for the Jacobian of the exponential map on left-trivialized Lie groups
    # J_r = I - 1/2 adₓ + 1/6 adₓ^2 - 1/24 adₓ^3 + 1/120 adₓ^4 - ...
    Numerator = I - 0.1 * ad_X + (1/60) * ad_X2
    Denominator = I + 0.4 * ad_X + 0.05 * ad_X2

    return Denominator \ Numerator
end

# jacobian_exp: the left trivialized Jacobian [Mahony, 2024, Theorem 4.3]
function jacobian_exp_PTC_2nd(G::AbstractLieGroup, p, Xp, b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis())
    # parallel transport on the canonical Cartan-Schouten connection
    adx = adjoint_algebra_matrix(G, -0.5 * Xp)
    P = exp(adx)
    # include 2nd order curvature correction
    return P * (LinearAlgebra.I + 1 / 6 * adx^2)
end

function jacobian_exp_PTC_4th(G::AbstractLieGroup, p, Xp, b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis())
    # parallel transport on the canonical Cartan-Schouten connection
    adx = adjoint_algebra_matrix(G, -0.5 * Xp)
    P = exp(adx)
    # include 4th order curvature correction
    return P * (LinearAlgebra.I + 1 / 6 * adx^2 + 1 / 120 * adx^4)
end

# does LieGroups provide a specialized (closed form) jacobian_exp! for this group and representation?
function _has_lie_jacobian_exp(G::AbstractLieGroup, g, X, b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis())
    return hasmethod(
        LieGroups.jacobian_exp!,
        Tuple{typeof(G), AbstractMatrix, typeof(g), typeof(X), typeof(b)},
    )
end

"""
    $SIGNATURES

Best available left-trivialized Jacobian of the Lie group exponential map,
selecting the implementation per group: delegates to the analytical closed form
`LieGroups.jacobian_exp` when a specialized method is available for group `G` (and the
representation of `X`), otherwise falls back to the fast [2,2]-Padé approximation
[`jacobian_exp_pade_22`](@ref).

Same argument convention as `LieGroups.jacobian_exp`.

See also: [`jacobian_exp_series`](@ref), [`jacobian_exp_pade_22`](@ref),
[`jacobian_exp_PTC_2nd`](@ref).
"""
function jacobian_exp_best(
    G::AbstractLieGroup,
    g,
    X,
    b::DefaultLieAlgebraOrthogonalBasis = DefaultLieAlgebraOrthogonalBasis(),
)
    # TODO use compile time dispatch rather runtime if 
    # TODO use compile time dispatch rather runtime if 
    if _has_lie_jacobian_exp(G, g, X, b)
        return LieGroups.jacobian_exp(G, g, X, b)
    end
    # numeric fallback when LieGroups has no closed form for this group
    return jacobian_exp_pade_22(G, g, X, b)
end

# NOTE no 2-arg convenience is provided: callers must supply a base point `g`. When no real
# point is available, reuse `X` itself as `g` (i.e. `jacobian_exp_best(G, X, X)`) -- this is
# base-point-independent and numerically verified equal to using a genuine point, and avoids
# an `identity_element` allocation. Do NOT use `Identity(G)` as a stand-in: as of LieGroups.jl
# 0.1.12, `LieGroups.jacobian_exp(G, Identity(G), X, b)` throws (`allocate_result` cannot infer
# an element type from the lazy `Identity` singleton) -- TODO fix upstream in LieGroups.jl.

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

See also: [`jacobian_exp_best`](@ref), [`jacobian_exp_series`](@ref), `Manifolds.parallel_transport_direction`, `Manifolds.parallel_transport_to`, `Manifolds.parallel_transport_along`
"""
parallel_transport_best(M::AbstractManifold, p, X::AbstractArray, d::AbstractArray) =
    Manifolds.parallel_transport_direction(M, p, X, d)

## ================================= Lie (a/A)djoints =================================
## ---------------------------------- Almost generic ----------------------------------

"""
    $SIGNATURES

Matrix of the *small adjoint* operator ``\\mathrm{ad}_X`` of a Lie group `M`, over the
default Lie algebra orthogonal basis. Input `X` is a Lie algebra tangent vector; column `j`
of the returned matrix holds the coordinates of the Lie bracket ``[X, E_j]`` over the
algebra basis ``E_j``, so that `A * c` gives the coordinates of ``\\mathrm{ad}_X`` applied
to a vector with coordinates `c`.

Generic fallback: LieGroups.jl provides only the bracket *operator* (`lie_bracket`), so the
matrix is assembled column by column by bracketing `X` against each basis element. Works on
any Lie group (including product manifolds). Group-specific, allocation-free closed-form
overrides are provided below for e.g. SO(3) and SE(2)/SE(3).

Notes
- Ref [Chirikjian 2012, Vol.2, pg.30, eq.10.59b]
- the *big (group) adjoint* counterpart is [`adjoint_group_matrix`](@ref).

See also: `LieGroups.lie_bracket`, [`adjoint_group_matrix`](@ref).
"""
function adjoint_algebra_matrix(M::AbstractLieGroup, X)
    𝔤 = LieAlgebra(M)
    dim = manifold_dimension(M)
    T = eltype(X)
    A = Matrix{T}(undef, dim, dim)
    e_j = zeros(T, dim)
    E_mat = zero(X)
    bracket = zero(X)
    for j in 1:dim
        e_j[j] = one(T)
        hat!(𝔤, E_mat, e_j)
        lie_bracket!(𝔤, bracket, X, E_mat)
        vee!(𝔤, view(A, :, j), bracket)
        e_j[j] = zero(T)
    end
    return A
end

"""
    $SIGNATURES

Matrix of the *big (group) adjoint* ``\\mathrm{Ad}_p`` of a Lie group `M`, over the default
Lie algebra orthogonal basis. Input `p` is a point on the group; the returned matrix maps
the coordinates of a Lie algebra vector `X` to the coordinates of
``\\mathrm{Ad}_p(X) = p X p^{-1}``.

Generic fallback: the adjoint matrix is the Jacobian of conjugation ``c_p(h) = p∘h∘p^{-1}``
at the identity, so this delegates to `LieGroups.jacobian_conjugate(M, p, Identity(M))`
(see the LieGroups.jl docs, and [SolaDerayAtchuthan:2021] where this Jacobian is called the
adjoint matrix). Group-specific, allocation-free closed-form overrides are provided below
for e.g. SO(3) and SE(2)/SE(3).

Notes
- the *small (algebra) adjoint* counterpart is [`adjoint_algebra_matrix`](@ref), with
  ``\\mathrm{Ad}_{\\exp(X)} = \\exp(\\mathrm{ad}_X)``.

See also: `LieGroups.jacobian_conjugate`, [`adjoint_algebra_matrix`](@ref).
"""
function adjoint_group_matrix(M::AbstractLieGroup, p)
    return LieGroups.jacobian_conjugate(M, p, Identity(M))
end

# closed forms, overriding the generic fallbacks above
# [Chirikjian, 2012 Vol2, p.39]
adjoint_algebra_matrix(::typeof(SpecialOrthogonalGroup(3)), X::AbstractMatrix) = X

adjoint_group_matrix(::typeof(SpecialOrthogonalGroup(3)), R::AbstractMatrix) = R

## For SO(3) Jr and Jl + inv closed forms, see [Chirikjian 2012, Vol2, Vol2 p.40] !!!
function adjoint_algebra_matrix(
    ::typeof(SpecialEuclideanGroup(2; variant = :right)), d::ArrayPartition
)
    Vx = SA[d.x[1][2]; -d.x[1][1]]
    Ω = d.x[2]
    return vcat(hcat(Ω, Vx), zero(SMatrix{1, 3, Float64}))
end

function adjoint_algebra_matrix(
    ::typeof(SpecialEuclideanGroup(3; variant = :right)), d::ArrayPartition
)
    v1x = LieGroups.hat(LieAlgebra(SpecialOrthogonalGroup(3)), d.x[1]) # skew(d.x[1])
    Ω = d.x[2]
    return vcat(hcat(Ω, v1x), hcat(zero(SMatrix{3, 3, Float64}), Ω))
end

function adjoint_group_matrix(
    ::typeof(SpecialEuclideanGroup(2; variant = :right)), p::ArrayPartition
)
    t = p.x[1]
    R = p.x[2]
    return vcat(hcat(R, -SA[0 -1; 1 0] * t), SA[0 0 1])
end

function adjoint_group_matrix(
    ::typeof(SpecialEuclideanGroup(3; variant = :right)), p::ArrayPartition
)
    t = p.x[1]
    R = p.x[2]
    st = LieGroups.hat(LieAlgebra(SpecialOrthogonalGroup(3)), t) # skew(t)
    return vcat(hcat(R, st * R), hcat(zero(SMatrix{3, 3, Float64}), R))
end
