
## =============================================================================
## Function Overloads

# NOTES, 
# - ManellicTree kernel types have mean and cov methods for easy access
# - ManellicTree currently only supports MvNormalKernel types


Statistics.mean(m::MvNormalKernel) = m.shim.params         # mean(m.p)
# Statistics.cov(m::MvNormalKernel) = cov(m.p)     # note also about m.sqrt_iΣ
Statistics.cov(m::MvNormalKernel) = m.shim.functional.Σ.mat # direct from stored matrix
# TODO, drop the diagm on std here
Statistics.std(m::MvNormalKernel) = diagm(std(m.shim.functional)) # sqrt(cov(m)) # regular sqrt (not of inverse)
# FIXME use MvNormal.pdmatrix for faster access to cov's Cholesky
sqrt_Σ(m::MvNormalKernel) = std(m)
sqrt_iΣ(m::MvNormalKernel) = cov(m) |> sqrt |> inv

getBW(mker::MvNormalKernel) = sqrt_Σ(mker) |> collect # cov(mker) |> collect


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
    if length(mean(ekr)) != length(p)
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
    if length(p) != length(q)
        @error "distanceMalahanobisCoordinates does not compute when manifold dimension differs from the input points, likely a partials mismatch earlier in the stack?" M partial p q maxlog=20
        # error("hold up")
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
    _μ(s::AbstractArray, _p::Tuple, pf::Function) = begin
        _s = deepcopy(s)
        _s_ = pf(_s)
        fill!(s, NaN)
        s_ = pf(s) # required for non-trivial points, eg SO/SE have more complicated representations
        s_ .= _s_ # copy back only the partials, leave NaNs in the rest
        return s
    end
        # _μ(s::AbstractVector, _p::Tuple, pf::Nothing) = begin
        #     # OBSOLETE DEV HACK, TODO delete
        #     _s = _forcemutable(s)
        #     _s[setdiff(1:length(s), _p)] .= NaN
        #     return _s
        # end
        # _μ(s::AbstractMatrix, _p::Tuple, pf::Nothing) = begin
        #     # OBSOLETE DEV HACK, TODO delete
        #     _s = _forcemutable(s)
        #     itr = setdiff(1:length(s), _p) 
        #     _s[itr, :] .= NaN
        #     _s[:, itr] .= NaN
        #     return _s
        # end

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


function MvNormalKernel{
    ApproxManifoldProducts.DensityKernel{
        L,
        MvNormal{F,P,Z},
        S
    }
}(
    src::MvNormalKernel;
    partl_cb::Union{Nothing, <:Function} = nothing, # partial is pulled from kernel...
    Σ = nothing,
) where {L,F,P,Z,S}

    _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
    _sap(::Type{ArrayPartition{T,_S}}) where {T,_S} = _S
    _new(s) = S(s)
    _new(s::ArrayPartition{T,O}) where {T,O} = ArrayPartition(begin
        S_ = _sap(S)
        [S_.parameters[i](v) for (i,v) in enumerate(s.x)]
    end...)

    m = _new(src.shim.params)
    Σ_ = isnothing(Σ) ? cov(src.shim.functional) : Σ

    MvNormalKernel(
        m,
        _matType(P)(Σ_),
        src.shim.weight;
        partial = L,
        partl_cb,
    )
end


function MvNormalKernel(
    src::MvNormalKernel{
        ApproxManifoldProducts.DensityKernel{
            L,
            MvNormal{F,P,Z},
            S
        }
    };
    partl_cb::Union{Nothing, <:Function} = nothing, # partial is pulled from kernel...
    Σ = nothing,
    kw...
) where {L,F,P,Z,S}
    return MvNormalKernel{ApproxManifoldProducts.DensityKernel{L,MvNormal{F,P,Z},S}}(src; partl_cb, Σ, kw...)
end


# case for identical types not requiring any conversions
function Base.convert(
    ::Type{T},
    src::T,
) where {T <: MvNormalKernel}
    return src
end


