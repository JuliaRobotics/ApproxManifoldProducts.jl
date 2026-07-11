# Common Utils

function calcProductGaussians_flat(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}}, # point type commonly known as P (actually on-manifold)
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    μ0 = nothing, #mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = nothing, #inv.(Σ_),
    weight::Real = 1.0,
    partials::Union{<:AbstractVector, <:Tuple} = [nothing for _ = 1:length(μ_)],
    do_transport_correction::Bool = true,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    # resolve partial reductions when summing "incomplete" inverse covariance matrices
    function _sumprecisionpartials(S)
        if all(isnothing.(partials))
            _S = +(S...)
            return _S, length(μ_)*ones(Int, size(S[1], 1))
        end
        s1 = _forcemutable(S[1])
        _S = similar(s1)
        fill!(_S, 0.0)
        # duplicating in new _mergepartials function, WIP
        prlm = zeros(Int, size(_S, 1))
        for (s, pl) in zip(S, partials)
            _S_ = _viewprl(_S, pl)
            _S_ .+= _viewprl(s, pl)
            if isnothing(pl)
                prlm .+= 1
            else
                for i in pl
                    prlm[i] += 1
                end
            end
        end
        # set any untouched precision variances to Inf
        imask = prlm .== 0
        if 0 < sum(imask)
            __S = view(_S, imask, imask)
            for i = 1:sum(imask)
                __S[i, i] = Inf
            end
        end
        # return summed precions and partialmask
        return _S, prlm
    end

    # _μ0
    # _Λ_
    _μ0 = isnothing(μ0) ? _mean(M, μ_; partials) : μ0
    _Λ_ = isnothing(Λ_) ? _invs(Σ_; partials) : Λ_
    # prepare an emply destination template matrix
    tmpl = _forcemutable(similar(_μ0))
    fill!(tmpl, 0)

    # calc sum of inv covariances while honoring partials
    Λ, prlm = _sumprecisionpartials(_Λ_)

    _Log(manif::AbstractLieGroup, μ, u) = vee(LieAlgebra(manif), log(manif, μ, u))
    _Log(manif::AbstractManifold, μ, u) = vee(manif, μ, log(manif, μ, u))

    # do the actual Guassian product while stepping around the partials
    # calc the covariance weighted delta means of incoming points and covariances
    ΛΔμc = mapreduce(+, zip(_Λ_, μ_, partials)) do (s, u, pl)
        if isnothing(pl)
            # @info "calcProductGaussians_flat" typeof(u) typeof(_μ0) typeof(s)
            Δuvee = _Log(M, _μ0, u)
            s * Δuvee
        else
            M_, rp_, fnc_ = getManifoldPartial(M, _makevec(pl))
            _μ0_ = fnc_(_μ0)
            _u_ = fnc_(u)
            _Δuvee = _Log(M_, _μ0_, _u_)
            tmp = deepcopy(tmpl)
            _tmp = _viewprl(tmp, pl)
            _s = _viewprl(s, pl)
            _tmp .= _s * _Δuvee
            tmp
        end
    end

    # prepare partial-aware product mean containers
    plmask = 0 .< prlm
    _Λ = view(Λ, plmask, plmask)
    _ΛΔμc = view(ΛΔμc, plmask)
    _Δμc = zeros(length(ΛΔμc))
    __Δμc = view(_Δμc, plmask)
    # in-place calculate the delta mean
    __Δμc .= _Λ \ _ΛΔμc

    Σr = inv(Matrix(Λ))
    for i in (1:length(prlm))[prlm .== 0]
        Σr[i, i] = Inf # likely better to have /Lambda have 0s on partials instead
    end
    # return the full dimension product mean and covariance (with honored partials)
    return _Δμc, Σr, prlm
end

"""
    $SIGNATURES

Calculate covariance weighted mean as product of incoming Gaussian points `μ_` and coordinate covariances `Σ_`.

Notes
- Return both weighted mean and new covariance (teh congruent product)
- More efficient helper function allows passing keyword inverse covariances `Λ_` instead. 
- Assume `size(Σ_[1],1) == manifold_dimension(M)`.
- calc lambdas first and use to calculate mean product second.
- https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
- Pennec, X. Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements, HAL Archive, 2011, Inria, France.

Keyword `jacobian_exp_fnc` selects the left-trivialized Jacobian of exp function, called as
`jacobian_exp_fnc(M, p, d)` with `p` the group point the Jacobian is evaluated relative to and
`d` a Lie algebra element:
- `jacobian_exp_best` (default): best available left-trivialized Jacobian of exp, i.e. the analytical
  closed form (via `LieGroups.jacobian_exp`) where available, otherwise a fast Padé fallback.
- `jacobian_exp_PTC_2nd` / `jacobian_exp_PTC_4th`: approximate via parallel transport on the canonical
  Cartan-Schouten connection with 2nd/4th order curvature correction.

DevNotes:
- TODO avoid recomputing covariance matrix inverses all the time -- work directly with Precision matrix and pull-back instead
"""
function calcProductGaussians(
    M::AbstractLieGroup,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}}, # point type commonly known as P (actually on-manifold)
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    μ0 = nothing, # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = nothing,
    partials::Union{<:AbstractVector, <:Tuple} = [nothing for _ = 1:length(μ_)],
    do_transport_correction::Bool = true,
    jacobian_exp_fnc = jacobian_exp_best,
    weight::Real = 1.0,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    𝔤 = LieAlgebra(M)

    # step 0, resolve partials
    # Tangent space reference around the evenly weighted mean of incoming points
    _μ0 = something(μ0, _mean(M, μ_; partials))
    _Λ_ = something(Λ_, _invs(Σ_; partials))

    # step 1, basic/naive Gaussian product (ignoring disjointed covariance coordinates) 
    Xc_μ0, Σn, prlm =
        calcProductGaussians_flat(M, μ_, Σ_; μ0 = _μ0, Λ_ = _Λ_, weight, partials)

    # correction on basis μ0 to account for the fact that the product mean is not actually at the tangent space origin (μ0) of the incoming covariances
    μ1 = exp(M, _μ0, hat(𝔤, Xc_μ0, P))

    # @info "calcProductGaussians" eltype(μ_) typeof(_μ0) typeof(μ1)

    # for development and testing cases return without doing transport
    # FIXME partials skips parallel transport correction #330
    do_transport_correction && all(isnothing.(partials)) ? nothing : (return μ1, Σn, prlm)

    # first transport (push forward) covariances to common coordinates (at μ1)
    Σμ1_hat = map(zip(μ_, Σ_)) do (p, Σp)
        Xμ1 = log(M, μ1, p)
        pJμ1 = jacobian_exp_fnc(M, μ1, Xμ1)
        μ1Jp = inv(pJμ1) # reminder, Xμ1 is the vector from μ1 to p and we want to push forward covariances to the first estimated mean μ1 and therefore take the inverse of the Jacobian here to push Σp forward from p to μ1.
        return μ1Jp * Σp * (μ1Jp') # Ge, Mahony 2024, eq. 10
    end

    # do product of transported covariances, relative to the identity element because of compose above
    # consider using Δμ in place of _μ0
    Xc_μ1, Σμ1_diam, prlm = ApproxManifoldProducts.calcProductGaussians_flat(
        M,
        μ_,
        Σμ1_hat;
        μ0 = μ1,
        weight,
        partials,
    ) # partials do not make it this far yet

    # workaround needed during IIF v0.37
    _P = typeof(_forcemutable(μ_[1]))

    # Reset step to absorb extended μ+ coordinates into kernel on-manifold μ 
    X_μ1 = hat(𝔤, Xc_μ1, _P) # FIXME, should be just P when hode static over in-place
    μplus = exp(M, μ1, X_μ1)
    μpJμ1 = jacobian_exp_fnc(M, μ1, X_μ1)
    Σμplus = μpJμ1 * Σμ1_diam * (μpJμ1')

    # return new mean and covariance
    return μplus, Σμplus, prlm
end

# Affie asks if we should consider isotropic or "piecewise" isotropic covariances 
#  to simplify this step, as the parallel transport of a full covariance matrix 
#  is expensive and may not be necessary for some applications -- i.e. only scalar transport.
# Dehann asks for homotopy density, bottom of tree associates with smallest eigen values,
#  so isotropic significance may be traceable.
# Part of using new name homotopy -- i.e. continuation from isotropic to full covariance depending on depth.
#  separation between leaf kernels infitesimally becomes zero curvature.
#  In the extreme case of infinite depth homotopy density tree, eigen values become zero so bandwidths become irrelevant.

# REMEMBER, this is an additional dispatch case for covariances passed as diagonal vectors 
function calcProductGaussians(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}},
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}},
    w...;
    partials::AbstractVector = [nothing for _ = 1:length(μ_)],
    kw...,
) where {N, P, S <: AbstractVector}
    # error("calcProductGaussians excessive wrapper?")
    # still pass nothing, to avoid stack overflow.  Only Λ_ is needed further
    if isnothing(eltype(partials))
        error("diagonal case for calcProductGaussian partial support is TODO")
    end

    kers = [
        ConcentratedGaussianKernel(p, C; partial) for
        (p, C, partial) in zip(μ_, Σ_, partials)
    ]
    return calcProductGaussians(M, kers, w...; kw...)
end

"""
    $SIGNATURES

EXPERIMENTAL: On-manifold product of Gaussians.

DevNotes
- FIXME do product of concentrated Gaussians on Lie group (approximation):
  - See Section 3.2 and 4 of [Ge, van Goor, Mahony: A Geometric Perspective on using Gaussian Distributions on Lie Groups, 2024].
  - Also see upstream utils, https://juliamanifolds.github.io/Manifolds.jl/stable/features/distributions.html
- FIXME is parallel transport needed when multiplying with covariances from difffent tangent spaces?
"""
function calcProductGaussians(
    M::AbstractManifold,
    kernels::Union{
        <:AbstractVector{<:ConcentratedGaussianKernel}, # FIXME, product of components with different partials???
        <:NTuple{N, <:ConcentratedGaussianKernel},
    };
    μ0 = nothing,
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
    jacobian_exp_fnc = jacobian_exp_best,
) where {N}
    _getmat(s::AbstractMatrix) = s

    # EXPERIMENTAL, product of partials
    # FIXME
    # M_ = __getprt(M)
    # μ_ = (s->__getprt(mean(s))).(kernels) # This is a ArrayPartition which IS DEFINITELY ON MANIFOLD (we dispatch on mean)
    # Σ_ = (s->__getprt(cov( s))).(kernels) # .|> s -> s.mat  # on tangent
    μ_ = mean.(kernels)
    Σ_ = (s->_getmat(cov(s))).(kernels) # on tangent
    partials = _getprl.(kernels)
    # CHECK this should be on-manifold for points
    # parallel transport needed for covariances from different tangent spaces
    _μ, _Σ, ipc = calcProductGaussians(
        M,
        μ_,
        Σ_;
        μ0,
        partials,
        do_transport_correction,
        jacobian_exp_fnc,
    )
    # @info "calcProductGaussians" typeof(μ_) typeof(_μ)

    # FIXME, inflate any partial results
    _partial = findall(!iszero, ipc)
    __partial = length(_partial) == manifold_dimension(M) ? nothing : _partial
    __partial_ = _tuple(__partial)
    M_, reprl, partl_cb = getManifoldPartial(M, __partial_, _μ)
    return ConcentratedGaussianKernel(_μ, _Σ, weight; partial = __partial_, partl_cb)
end
