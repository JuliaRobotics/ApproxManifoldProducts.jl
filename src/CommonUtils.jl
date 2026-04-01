# Common Utils

_forcemutable(s::MMatrix) = s
_forcemutable(s::AbstractMatrix) = MMatrix{size(s)...}(s)
_forcemutable(s::MVector) = s
_forcemutable(s::AbstractVector) = MVector{length(s)}(s)

# kernels explicitly change to partial definition via tuples (for clarity during development) 
_tuple(p::Nothing) = p
_tuple(p::Tuple) = p
_tuple(p::AbstractVector{<:Integer}) = tuple(p...)

_makevec(w::AbstractVector) = w
_makevec(w::Tuple) = [w...]

_getprl(::MvNormalKernel{<:DensityKernel{partial}}) where partial = partial
_getpartial(  ::Nothing, s) = s
_getpartial(_pr::Tuple, v::AbstractVector) = view(v, SVector(_pr...))
_getpartial(_pr::Tuple, v::AbstractMatrix) = view(v, SVector(_pr...), SVector(_pr...))
_getpartial(_pr::Tuple, m::AbstractManifold) = getManifoldPartial(m, _makevec(_pr))[1]
_getpartial(partial::AbstractVector{<:Int}, s) = _getpartial(_tuple(partial), s)

_viewprl(s::AbstractArray, partial::Nothing) = s
_viewprl(s::AbstractArray, partial::Tuple) = _viewprl(s, _makevec(partial))
_viewprl(s::AbstractVector, partial::AbstractVector) = view(s, partial)
_viewprl(s::AbstractMatrix, partial::AbstractVector) = view(s, partial, partial)


function _invs(
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}}; 
    partials::Union{<:AbstractVector, <:Tuple}
) where {N, S <: AbstractMatrix{<:Real}}
    d = size(Σ_[1])[1]
    infs = diagm(MVector{d}([Inf for _ in 1:d]))
    Λs = [deepcopy(infs) for _ in 1:length(Σ_)]
    for (i,s) in enumerate(Σ_)
        dst = _viewprl(Λs[i], partials[i]) 
        dst .= inv(_viewprl(s, partials[i]))
    end
    return Λs
end

function _mean(
    M::AbstractManifold, 
    v::Union{<:AbstractVector{P}, <:NTuple{N, P}}; 
    partials::Union{<:AbstractVector, <:Tuple}
) where {N, P <: AbstractArray}
    # hack during dev testing
    if all(isnothing.(partials))
        return mean(M, _makevec(v))
    elseif P <: AbstractVector
        d = manifold_dimension(M)
        mn = MVector{d}([0.0 for _ in 1:d])
        cu = MVector{d}([0 for _ in 1:d])
        for (s,pl) in zip(v,partials)
            _mn = _viewprl(mn, pl)
            _mn .+= _viewprl(s, pl)
            _cu = _viewprl(cu, pl)
            _cu .+= 1
        end
        return mn ./ cu
    else
        error("TODO calc partial mean of non-vector manifold types $(M), v isa $(typeof(v)), given $(partials)")
    end
end

"""
    $SIGNATURES

A clunky repeat calculation of one product kernel.
"""
function updateProductSample(
    dest::BallTreeDensity,
    proposals::Vector{BallTreeDensity},
    manifolds::Tuple,
    smplIdx::Int,
    labels::Vector{Int},
)
    #

    Ndens = length(proposals)
    Ndim = Ndim(dest)

    densLen = Npts.(proposals)

    calclambdas = zeros(Ndim, Ndens)
    calcmu = zeros(Ndim, Ndens)
    destMu = zeros(Ndim)
    destCov = 0.0

    @inbounds @fastmath @simd for dim = 1:Ndim
        for j = 1:Ndens
            calclambdas[dim, j] = 1.0 / getBW(proposals[j])[dim, labels[j]]
            calcmu[dim, j] = getPoints(proposals[j])[dim, labels[j]]
        end
        destCov = getLambda(calclambdas)
        destCov = 1.0 / destCov
        # μ = 1/Λ * Λμ  ## i.e. already scaled to mean only
        destMu[dim] = getMu(calcmu[dim, :], calclambdas[dim, :], destCov)
    end

    # previous points
    pts = getPoints(dest)
    pts[:, smplIdx] = destMu

    return manikde!(pts, manifolds)
end

# TODO this should be a public method relating to getManifold
function _getManifoldFullOrPart(mkd::ManifoldKernelDensity, aspartial::Bool = true)
    if aspartial && isPartial(mkd)
        getManifoldPartial(mkd.manifold, mkd._partial)
    else
        mkd.manifold
    end
end



function calcProductGaussians_flat(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}}, # point type commonly known as P (actually on-manifold)
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    μ0 = nothing, #mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = nothing, #inv.(Σ_),
    weight::Real = 1.0,
    partials::Union{<:AbstractVector, <:Tuple} = [nothing for _ in 1:length(μ_)],
    do_transport_correction::Bool = true,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    # resolve partial reductions when summing "incomplete" inverse covariance matrices
    function _sumprecisionpartials(S)
        if all(isnothing.(partials))
            _S = +(S...)
            return _S, length(μ_)*ones(Int,size(S[1],1))
        end
        s1 = _forcemutable(S[1])
        _S = similar(s1)
        fill!(_S, 0.0)
        prlm = zeros(Int,size(_S,1))
        for (s,pl) in zip(S,partials)
            _S_ = _viewprl(_S, pl)
            # @show _S_ pl
            _S_ .+= _viewprl(s, pl)
            for i in pl
                prlm[i] += 1
            end
        end
        # set any untouched precision variances to Inf
        imask = prlm .== 0
        __S = view(_S, imask, imask)
        for i in 1:length(sum(imask))
            __S[i,i] = Inf
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

    # do the actual Guassian product while stepping around the partials
    # calc the covariance weighted delta means of incoming points and covariances
    ΛΔμc = mapreduce(+, zip(_Λ_, μ_, partials)) do (s, u, pl)
        if isnothing(pl)
            Δuvee = vee(LieAlgebra(M), log(M, _μ0, u))
            s * Δuvee
        else
            M_, rp_, fnc_ = getManifoldPartial(M, _makevec(pl))
            _μ0_ = fnc_(_μ0)
            _u_ = fnc_(u)
            _Δuvee = vee(LieAlgebra(M_), log(M_, _μ0_, _u_))
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
        Σr[i,i] = Inf # likely better to have /Lambda have 0s on partials instead
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

DevNotes:
- FIXME is parallel transport needed as products involve covariances from different tangent spaces?
- TODO avoid recomputing covariance matrix inverses all the time
"""
function calcProductGaussians(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}}, # point type commonly known as P (actually on-manifold)
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    μ0 = nothing, # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = nothing,
    partials::Union{<:AbstractVector, <:Tuple} = [nothing for _ in 1:length(μ_)],
    do_transport_correction::Bool = true,
    weight::Real = 1.0,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    # step 0, resolve partials
    # Tangent space reference around the evenly weighted mean of incoming points

    _μ0 = isnothing(μ0) ? _mean(M, μ_; partials) : μ0
    _Λ_ = isnothing(Λ_) ? _invs(Σ_; partials) : Λ_ # FIXME resolve partials issues 

    # step 1, basic/naive Gaussian product (ignoring disjointed covariance coordinates) 
    Δμn, Σn, prlm = calcProductGaussians_flat(M, μ_, Σ_; μ0=_μ0, Λ_=_Λ_, weight, partials)
    # correction on basis μ0 to account for the fact that the product mean is not actually at the tangent space origin (μ0) of the incoming covariances
    Δμ = exp(M, _μ0, hat(M, _μ0, Δμn))

    # for development and testing cases return without doing transport
    # FIXME partials skips parallel transport correction #330
    do_transport_correction && all(isnothing.(partials)) ? nothing : (return Δμ, Σn, prlm)

    # first transport (push forward) covariances to common coordinates
    # see [Ge, van Goor, Mahony, 2024]
    iΔμ = inv(M, Δμ)
    μi_ = map(u -> LieGroups.compose(M, iΔμ, u), μ_)
    μi_̂ = map(u -> log(M, _μ0, u), μi_)
    # μi = map(u->vee(M,_μ0,u), μi_̂ )
    Ji = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie.(Ref(M), μi_̂)
    iJi = inv.(Ji)
    # Affie asks if we should consider isotropic or "piecewise" isotropic covariances 
    #  to simplify this step, as the parallel transport of a full covariance matrix 
    #  is expensive and may not be necessary for some applications -- i.e. only scalar transport.
    # Dehann asks for homotopy density, bottom of tree associates with smallest eigen values,
    #  so isotropic significance may be traceable.
    # Part of using new name homotopy -- i.e. continuation from isotropic to full covariance depending on depth.
    #  separation between leaf kernels reduces to zero curvature.
    #  In the extreme case of infinite depth homotopy density tree, eigen values are zero and bandwidths are isotropic. 
    Σi_hat = map((J, S) -> J * S * (J'), iJi, Σ_)

    # Reset step to absorb extended μ+ coordinates into kernel on-manifold μ 
    # consider using Δμ in place of _μ0
    Δμplusc, Σdiam, prlm =
        ApproxManifoldProducts.calcProductGaussians_flat(M, μi_, Σi_hat; μ0=_μ0, weight, partials) # partials do not make it this far yet
    Δμplus_̂ = hat(M, _μ0, Δμplusc)
    Δμplus = exp(M, _μ0, Δμplus_̂)
    μ_plus = LieGroups.compose(M, Δμ, Δμplus)
    Jμ = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, Δμplus_̂)
    Σ_plus = Jμ * Σdiam * (Jμ')

    # return new mean and covariance
    return μ_plus, Σ_plus, prlm
end

# REMEMBER, this is an additional dispatch case for covariances passed as diagonal vectors 
function calcProductGaussians(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}},
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}},
    w...;
    partials::AbstractVector = [nothing for _ in 1:length(μ_)],
    kw...,
) where {N, P, S <: AbstractVector}
    # error("calcProductGaussians excessive wrapper?")
        # still pass nothing, to avoid stack overflow.  Only Λ_ is needed further
    if isnothing(eltype(partials))
        error("diagonal case for calcProductGaussian partial support is TODO")
    end

    kers = [MvNormalKernel(p, C; partial) for (p, C, partial) in zip(μ_, Σ_, partials)]
    return calcProductGaussians(
        M,
        kers,
        w...;
        kw...
    )
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
        <:AbstractVector{<:MvNormalKernel{<:DensityKernel}}, # FIXME, product of components with different partials???
        <:NTuple{N, <:MvNormalKernel{<:DensityKernel}}
    };
    μ0 = nothing,
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
) where {N}
    # __getprt(s) = _getpartial(partial, s)
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
    _μ, _Σ, ipc = calcProductGaussians(M, μ_, Σ_; μ0, partials, do_transport_correction)
    
    _tuplenothing(s::Nothing) = s
    _tuplenothing(v::AbstractVector) = tuple(v...)
    
    # FIXME, inflate any partial results
    _partial = findall(!iszero, ipc)
    __partial = length(_partial) == manifold_dimension(M) ? nothing : _partial
    return MvNormalKernel(_μ, _Σ, weight; partial=_tuplenothing(__partial))
end



#
