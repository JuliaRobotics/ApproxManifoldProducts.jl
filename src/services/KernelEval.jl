
## =============================================================================
## Function Overloads

# NOTES, 
# - ManellicTree kernel types have mean and cov methods for easy access
# - ManellicTree currently only supports MvNormalKernel types

Statistics.mean(m::MvNormalKernel) = m.shim.params         # mean(m.p)
# Statistics.cov(m::MvNormalKernel) = cov(m.p)     # note also about m.sqrt_iΣ
Statistics.cov(m::MvNormalKernel) = m.shim.functional.Σ.mat # direct from stored matrix
Statistics.std(m::MvNormalKernel) = sqrt(cov(m)) # regular sqrt (not of inverse)
# FIXME use MvNormal.pdmatrix for faster access to cov's Cholesky
sqrt_iΣ(m::MvNormalKernel) = cov(m) |> sqrt |> inv

function Base.show(
    io::IO, 
    mvk::MvNormalKernel{<:DensityKernel{partial}}
) where partial
    μ = mean(mvk)
    Σ2 = cov(mvk)
    # Σ=sqrt(Σ2)
    d = size(Σ2, 1)
    print(io, "MvNormalKernel")
    print(io, "(d=", d)
    print(io, isnothing(partial) ? "" : "*->$partial")
    print(io, ",μ=", round.(μ; digits = 3))
    print(io, ",Σ^2=[", round(Σ2[1]; digits = 3))
    if 1 < d
        print(io, "...")
    end
    # det(T-I) is a proxy through volume meaure of Transform from unit covariance matrix to this instance
    # i.e. how large or rotated is this covariance instance
    println(
        io,
        "]); det(T-I)=",
        round(det(covTransformNormalized(Σ2) - diagm(ones(d))); digits = 3),
    )
    # ; det(Σ)=",round(det(Σ);digits=3), "
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", mvk::MvNormalKernel) = show(io, mvk)


## =============================================================================
## Various kernel accessors and functions

# also makes static
function projectSymPosDef(c::AbstractMatrix)
    s = size(c)
    # pretty fast to make or remake isbitstype from matrix
    _c = SMatrix{s...}(c)
    #TODO likely not intended project here: see AMP#283
    return issymmetric(_c) ? _c : project(Manifolds.SymmetricPositiveDefinite(s[1]), _c, _c)
end


function updateKernelBW(k::MvNormalKernel, _bw; sqrt_iΣ = inv(sqrt(_bw)))
    p = MvNormal(_bw)
    # sqrt_iΣ_ = typeof(k.sqrt_iΣ)(sqrt_iΣ)
    return MvNormalKernel(mean(k), _bw, k.shim.weight)
end
updateKernelBW(ekr::MvNormalKernel, ::Nothing) = ekr # avoid ifs for noops

function evaluate(
    M::AbstractManifold,
    ekr::MvNormalKernel{<:DensityKernel{partial}},
    p, # on manifold point
) where partial
    _manidim(::Nothing) = manifold_dimension(M)
    _manidim(::Tuple) = _manidim(nothing) - length(partial)
    dim_ = _manidim(partial)
    cov_ = _getpartial(partial, cov(ekr))
    nscl = 1 / sqrt((2 * pi)^dim_ * det(cov_))
    # @info "evaluate kernel" partial dim_ nscl
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
    return R * S
end


function distanceMalahanobisCoordinates(
    M::AbstractManifold,
    K::AbstractKernel,
    q,
    basis = DefaultOrthogonalBasis(),
)
    p = mean(K)
    i_p = inv(M, p)
    pq = LieGroups.compose(M, i_p, q)
    ϵ = identity_element(M, typeof(q))
    X = log(M, ϵ, pq)
    Xc = get_coordinates(M, ϵ, X, basis)
    # FIXME do partials like in AbstractLieGroup version
    return sqrt_iΣ(K) * Xc
end

function distanceMalahanobisCoordinates(
    M::AbstractLieGroup,
    K::AbstractKernel,
    q,
    _basis = nothing,
)
    p = mean(K)
    i_p = inv(M, p)
    pq = LieGroups.compose(M, i_p, q)
    X = log(M, pq)
    Xc = vee(LieAlgebra(M), X)
    return sqrt_iΣ(K) * Xc
end

function distanceMalahanobisSq(
    M::AbstractManifold,
    K::AbstractKernel,
    q,
    basis = DefaultOrthogonalBasis(),
)
    δc = distanceMalahanobisCoordinates(M, K, q, basis)
    # return inner(M, p, X, X) # did not work as inner gave almost 2x the answer?
    return δc' * δc
end

function distanceMalahanobisSq(
    M::AbstractLieGroup,
    K::AbstractKernel,
    q,
    # basis=DefaultOrthogonalBasis()
)
    δc = distanceMalahanobisCoordinates(M, K, q)
    # return inner(M, p, X, X) # did not work as inner gave almost 2x the answer?
    return δc' * δc
end

function _distance(
    M::AbstractManifold,
    p::AbstractVector,
    q::AbstractVector,
    kernel = (_p) -> MvNormalKernel(;
        p = MvNormal(_p, SVector(ntuple((s) -> 1, manifold_dimension(M))...)),
    ),
    distFnc::Function = distanceMalahanobisSq,
)
    return distFnc(M, kernel(p), q)
end

"""
$SIGNATURES

Normal kernel used for Hilbert space embeddings.
"""
ker(
    M::AbstractManifold,
    p,
    q,
    sigma::Real = 0.001,
    distFnc::Function = (_M, _p, _q) -> distance(_M, _p, _q)^2,
) = exp(-sigma * distFnc(M, p, q)) # _distance(M,p,q) # 
