
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

    #FIXME ON FIRE, confirm points from kernel match the manifold in presence of partials, else scale cant be guaranteed
    if length(mean(ekr)) != manifold_dimension(M)
        @error "FIXME, trying to evaluate kernel with mean of different dimension than the manifold, isa partials/marginal?" M partial mean(ekr) p maxlog=20
    end

    # ASSUMPTION, MvNormalKernel is always full dim and is reduced during computation to any partial/marginal info
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

    partial = _getprl(K)
    _Xc = _getpartial(partial, Xc)
    s_iΣ = _sqrt_iΣ(K)
    return s_iΣ * _Xc
    # return sqrt_iΣ(K) * Xc
end

function distanceMalahanobisCoordinates(
    M::AbstractLieGroup,
    K::MvNormalKernel{<:DensityKernel{partial}},
    q,
    _basis = nothing,
) where partial
    
    M_, repr, cb  = getManifoldPartial(M, partial)
    # 26Q2, super important, kernel K as partial, mean should be full dimensional and possible NaNs off-partial
    p = mean(K)
    if length(p) != manifold_dimension(M) || length(q) != manifold_dimension(M)
        @error "distanceMalahanobisCoordinates does not compute when manifold dimension differs from the input points, likely a partials mismatch earlier in the stack?" M partial p q maxlog=20
    end
    
    i_p = inv(M, p)
    pq = LieGroups.compose(M, i_p, q)
    X = log(M, pq)
    Xc = vee(LieAlgebra(M), X)
    _Xc = _getpartial(partial, Xc)
    s_iΣ = _sqrt_iΣ(K)
    return s_iΣ * _Xc
end

function distanceMalahanobisSq(
    M::Union{<:AbstractManifold, <:AbstractLieGroup},
    K::AbstractKernel,
    q,
    basis = DefaultOrthogonalBasis(),
)
    δc = distanceMalahanobisCoordinates(M, K, q, basis)
    # return inner(M, p, X, X) # did not work as inner gave almost 2x the answer?
    return δc' * δc
end

# function distanceMalahanobisSq(
#     M::AbstractLieGroup,
#     K::AbstractKernel,
#     q,
#     basis=DefaultOrthogonalBasis();
#     partial::Union{Nothing,<:Tuple} = nothing,
# )
#     δc = distanceMalahanobisCoordinates(M, K, q)
#     # return inner(M, p, X, X) # did not work as inner gave almost 2x the answer?
#     return δc' * δc
# end

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



## ======================= LEGACY BELOW ==================================


# import Base: getproperty
# function Base.getproperty(k::MvNormalKernel, f::Symbol)
#     if f === :sqrt_iΣ
#         # super slow and hacky, but only legacy.  WIP replacing
#         cov(k) |> inv |> sqrt
#     else
#         return getproperty(k.shim, f)
#     end
# end


function MvNormalKernel(
    μ::AbstractArray, 
    σ::AbstractArray, 
    weight::Real = 1.0;
    partial::Union{Nothing, <:Tuple} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
)
    @warn "MvNormalKernel is deprecated, use ConcentratedGaussianKernel instead, barr partial [maxlog=10]" maxlog=10
    _μ(s::AbstractArray, _p::Nothing, pf::Union{Nothing, <:Function}) = s
    _μ(s::AbstractVector, _p::Tuple, pf::Nothing) = begin
        # FIXME, cannot assume straight coordinate partial indexing works for all array{1}'s
        _s = _forcemutable(s)
        _s[setdiff(1:length(s), _p)] .= NaN
        return _s
    end
    _μ(s::AbstractMatrix, _p::Tuple, pf::Nothing) = begin
        # HACK BY ASSUMING CALLER SOLVED MATRIX CASE??? OR FUNCTION DISPATCH???
        _s = _forcemutable(s)
        # FIXME ON FIRE, this does not work for Matrices!!!!
        itr = setdiff(1:length(s), _p) 
        _s[itr, :] .= NaN
        _s[:, itr] .= NaN
        return _s
    end
    _μ(s::AbstractArray, _p::Tuple, pf::Function) = pf(s) # required for non-trivial points, eg SO/SE have more complicated representations
    c_(s::AbstractMatrix, _p::Nothing) = s
    c_(s::AbstractVector, _p::Nothing) = diagm(s)    
    c_(s::AbstractMatrix, _p::Tuple) = _partialCovToDefault!(_p,_forcemutable(s))
    c_(s::AbstractVector, _p::Tuple) = diagm(_partialCovToDefault!(_p,_forcemutable(s)))
    # TODO _forcestatic
    Σ = c_(σ, partial)
    _c = projectSymPosDef(Σ)
    return MvNormalKernel(
        ConcentratedGaussianKernel(;
            weight = float(weight),
            p = _μ(μ, partial, partl_cb),
            covmat = _c, # cov(MvNormal(_c)),
            partial,
        )
    )
end


MvNormalKernel(; μ, p::MvNormal, weight = 1.0, partial = nothing, kw...) = MvNormalKernel(μ, cov(p), weight; partial, kw...)


function convert(
    ::Type{MvNormalKernel{
        ApproxManifoldProducts.DensityKernel{
            L,
            MvNormal{F,P,Z},
            S
        }
    }},
    src::MvNormalKernel,
) where {L,F,P,Z,S}

    _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
    _sap(::Type{ArrayPartition{T,_S}}) where {T,_S} = _S
    _new(s) = S(s)
    _new(s::ArrayPartition{T,O}) where {T,O} = ArrayPartition(begin
        S_ = _sap(S)
        [S_.parameters[i](v) for (i,v) in enumerate(s.x)]
    end...)

    m = _new(src.shim.params)

    MvNormalKernel(
        m,
        _matType(P)(cov(src.shim.functional)),
        src.shim.weight;
        partial = L
    )
end


# function MvNormalKernel(
#     μ::AbstractArray, 
#     σ::AbstractArray, 
#     weight::Real = 1.0
# )
#     c_(s::AbstractMatrix) = s
#     c_(s::AbstractVector) = diagm(s)
#     Σ = c_(σ)
#     _c = projectSymPosDef(Σ)
#     p = MvNormal(_c)
#     # NOTE, TBD, why not sqrt(inv(p.Σ)), this had an issue seemingly internal to PDMat.chol which breaks an already forced SymPD matrix to again be not SymPD???
#     sqrt_iΣ = sqrt(inv(_c))
#     return MvNormalKernel(; μ, p, sqrt_iΣ, weight = float(weight))
# end



# case for identical types not requiring any conversions
function Base.convert(
    ::Type{T},
    src::T,
) where {T <: MvNormalKernel}
    return src
end


# # case for different types requiring conversion
# function Base.convert(
#     ::Type{MvNormalKernel{T}},
#     src::MvNormalKernel,
# ) where {T}
#     #
#     _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
#     μ = convert(P, src.μ) # P(src.μ)
#     p = MvNormal(_matType(M)(cov(src.p)))
#     # sqrt_iΣ = iM(src.sqrt_iΣ)
#     return MvNormalKernel(μ, p, src.weight)
# end