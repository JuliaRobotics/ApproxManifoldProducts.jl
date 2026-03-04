# Common Utils

_forcemutable(s::MMatrix) = s
_forcemutable(s::AbstractMatrix) = MMatrix{size(s)...}(s)
_forcemutable(s::MVector) = s
_forcemutable(s::AbstractVector) = MVector{length(s)}(s)

# kernels explicitly change to partial definition via tuples (for clarity during development) 
_tuple(p::Nothing) = p
_tuple(p::AbstractVector{<:Integer}) = tuple(p...)

_getpartial(::MvNormalKernel{<:DensityKernel{partial}}) where partial = partial
_getpartial(  ::Nothing, s) = s
_getpartial(_pr::Tuple, v::AbstractVector) = view(v, SVector(_pr...))
_getpartial(_pr::Tuple, v::AbstractMatrix) = view(v, SVector(_pr...), SVector(_pr...))
_getpartial(_pr::Tuple, m::AbstractManifold) = getManifoldPartial(m, [_pr...])[1]
_getpartial(partial::AbstractVector{<:Int}, s) = _getpartial(_tuple(partial), s)


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



_makevec(w::AbstractVector) = w
_makevec(w::Tuple) = [w...]

function calcProductGaussians_flat(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}}, # point type commonly known as P (actually on-manifold)
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    μ0 = mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = inv.(Σ_),
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    # calc sum of covariances  
    Λ = +(Λ_...)

    # calc the covariance weighted delta means of incoming points and covariances
    ΛΔμc = mapreduce(+, zip(Λ_, μ_)) do (s, u)
        Δuvee = vee(LieAlgebra(M), log(M, μ0, u))
        s * Δuvee
    end

    # calculate the delta mean
    Δμc = Λ \ ΛΔμc

    return Δμc, inv(Λ)
end

"""
    $SIGNATURES

Calculate covariance weighted mean as product of incoming Gaussian points ``μ_`` and coordinate covariances ``Σ_``.

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
    μ0 = mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
    Λ_ = inv.(Σ_),
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
) where {N, P <: AbstractArray, S <: AbstractMatrix{<:Real}}
    # step 1, basic/naive Gaussian product (ignoring disjointed covariance coordinates) 
    Δμn, Σn = calcProductGaussians_flat(M, μ_, Σ_; μ0, Λ_, weight)
    Δμ = exp(M, μ0, hat(M, μ0, Δμn))

    # for development and testing cases return without doing transport
    do_transport_correction ? nothing : (return Δμ, Σn)

    # first transport (push forward) covariances to common coordinates
    # see [Ge, van Goor, Mahony, 2024]
    iΔμ = inv(M, Δμ)
    μi_ = map(u -> LieGroups.compose(M, iΔμ, u), μ_)
    μi_̂ = map(u -> log(M, μ0, u), μi_)
    # μi = map(u->vee(M,μ0,u), μi_̂ )
    Ji = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie.(Ref(M), μi_̂)
    iJi = inv.(Ji)
    Σi_hat = map((J, S) -> J * S * (J'), iJi, Σ_)

    # Reset step to absorb extended μ+ coordinates into kernel on-manifold μ 
    # consider using Δμ in place of μ0
    Δμplusc, Σdiam =
        ApproxManifoldProducts.calcProductGaussians_flat(M, μi_, Σi_hat; μ0, weight)
    Δμplus_̂ = hat(M, μ0, Δμplusc)
    Δμplus = exp(M, μ0, Δμplus_̂)
    μ_plus = LieGroups.compose(M, Δμ, Δμplus)
    Jμ = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, Δμplus_̂)
    Σ_plus = Jμ * Σdiam * (Jμ')

    # return new mean and covariance
    return μ_plus, Σ_plus
end

# additional support case where covariances are passed as diagonal-only vectors 
# still pass nothing, to avoid stack overflow.  Only Λ_ is needed further
function calcProductGaussians(
    M::AbstractManifold,
    μ_::Union{<:AbstractVector{P}, <:NTuple{N, P}},
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}};
    dim::Integer = manifold_dimension(M),
    Λ_ = map(s -> diagm(1.0 ./ s), Σ_),
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
) where {N, P, S <: AbstractVector}
    return calcProductGaussians(
        M,
        [MvNormalKernel(p, C) for (p, C) in zip(μ_, Σ_)];
        # dim, 
        do_transport_correction,
    ) # , Λ_
end # , Λ_
#

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
        <:AbstractVector{<:MvNormalKernel{<:DensityKernel{partial}}}, 
        <:NTuple{N, <:MvNormalKernel{<:DensityKernel{partial}}}
    };
    μ0 = nothing,
    weight::Real = 1.0,
    do_transport_correction::Bool = true,
) where {N, partial}

    __getprt(s) = _getpartial(partial, s)

    # EXPERIMENTAL, product of partials
    M_ = __getprt(M)
    μ_ = (s->__getprt(mean(s))).(kernels) # This is a ArrayPartition which IS DEFINITELY ON MANIFOLD (we dispatch on mean)
    Σ_ = (s->__getprt(cov( s))).(kernels) # .|> s -> s.mat  # on tangent
    # CHECK this should be on-manifold for points

    # parallel transport needed for covariances from different tangent spaces
    _μ, _Σ = if isnothing(μ0)
        calcProductGaussians(M_, μ_, Σ_; do_transport_correction)
    else
        calcProductGaussians(M_, μ_, Σ_; μ0, do_transport_correction)
    end

    return MvNormalKernel(_μ, _Σ, weight)
end



#
