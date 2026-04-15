# legacy content to facilitate transition to AMP


function resid2DLinear(μ, mus, Lambdas; diffop::Function = -)  # '-' exploits EuclideanManifold commutativity a-b = b-a
    # dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
    # @show round.(dμ, digits=4)
    # ret = sum( Lambdas.*dμ )
    r = map((mu, lam) -> diffop(μ[], mu) * lam, mus, Lambdas)
    return sum(r)
end

function solveresid2DLinear!(res, x, mus, Lambdas; diffop::Function = -)::Nothing
    res[1] = resid2DLinear(x, mus, Lambdas; diffop = diffop)
    return nothing
end

# import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function = -)::Float64
    solveresid2DLinear!(res, x, mus, Lambdas; diffop = diffop)
    return res[1]
end


function _update!(dst::MN, src::MN) where {MN <: ManifoldKernelDensity}
    KDE._update!(dst.belief, src.belief)
    @assert dst._partial == src._partial "AMP._update! can only be done for exactly the same ._partial values in dst and src"
    setPointsMani!(dst._u0, src._u0)
    dst.infoPerCoord .= src.infoPerCoord

    return dst
end


function _reducePartialManifoldElements(el::Symbol)
    if el == :Euclid
        return TranslationGroup(1)
    elseif el == :Circular
        return Circle()
    end
    return error("unknown manifold_symbol $el")
end

"""
    $SIGNATURES

Lots to do here, see RoME.jl #244 and standardized usage with Manifolds.jl.

Notes
- diffop( test, reference )   <===>   ΔX = inverse(test) * reference

DevNotes
- FIXME replace with Manifolds.jl #41, RoME.jl #244
"""
function buildHybridManifoldCallbacks(manif::Tuple)
    # TODO use multiple dispatch instead -- will be done for second version of system
    addopT = []
    diffopT = []
    getManiMu = []
    getManiLam = []

    for mn in manif
        if mn == :Euclid
            push!(addopT, +)
            push!(diffopT, -)
            push!(getManiMu, KDE.getEuclidMu)
            push!(getManiLam, KDE.getEuclidLambda)
        elseif mn == :Circular
            push!(addopT, addtheta)
            push!(diffopT, difftheta)
            push!(getManiMu, getCircMu)
            push!(getManiLam, getCircLambda)
        else
            error("Unrecognized manifold $(mn)")
        end
    end

    return (addopT...,), (diffopT...,), (getManiMu...,), (getManiLam...,)
end

# FIXME TO BE REMOVED
# _MtoSymbol(::Euclidean{Tuple{1}}) = :Euclid
# _MtoSymbol(::Circle) = :Circular

function _manifoldtuple(M::AbstractManifold)
    # TODO WIP to remove convert(Tuple, M) type piracy.
    # Also easier dev experience without so many convert (800+) and implicit conversion.
    @warn "Please use `_manifoldtuple(M)` instead. This will be removed (hopefully soon). Got" typeof(
        M,
    )
    return convert(Tuple, M)
end
# _manifoldtuple(M::ProductManifold) = _MtoSymbol.(M.manifolds)
# _manifoldtuple(M::Manifolds.TranslationGroup) = tuple([:Euclid for i in 1:manifold_dimension(M)]...)
function _manifoldtuple(M::LieGroups.TranslationGroup)
    return tuple([:Euclid for i = 1:manifold_dimension(M)]...)
end
_manifoldtuple(::typeof(LieGroups.CircleGroup(ℝ))) = (:Circular,)
function _manifoldtuple(
    ::LieGroup{ℂ, AbelianMultiplicationGroupOperation, Manifolds.Circle{ℂ}},
)
    return (:Euclid,)
end
_manifoldtuple(M::ValidationLieGroup) = _manifoldtuple(M.lie_group)

function _manifoldtuple(::Manifolds.Euclidean{Tuple{N}, ℝ}) where {N}
    return tuple([:Euclid for i = 1:N]...)
end
# _manifoldtuple(::Manifolds.Circle{ℝ})  = error("#FIXME")#(:Circular,)
# _manifoldtuple(::Manifolds.RealCircleGroup)  = (:Circular,)

_manifoldtuple(::typeof(Euclid)) = (:Euclid,)
_manifoldtuple(::typeof(Euclid2)) = (:Euclid, :Euclid)
_manifoldtuple(::typeof(Euclid3)) = (:Euclid, :Euclid, :Euclid)
_manifoldtuple(::typeof(Euclid4)) = (:Euclid, :Euclid, :Euclid, :Euclid)

_manifoldtuple(::typeof(SpecialOrthogonalGroup(2))) = (:Circular,)
_manifoldtuple(::typeof(SpecialOrthogonalGroup(3))) = (:Circular, :Circular, :Circular)
function _manifoldtuple(::typeof(SpecialEuclideanGroup(2; variant = :right)))
    return (:Euclid, :Euclid, :Circular)
end
function _manifoldtuple(::typeof(SpecialEuclideanGroup(3; variant = :right)))
    return (:Euclid, :Euclid, :Euclid, :Circular, :Circular, :Circular)
end
function _manifoldtuple(::typeof(TranslationGroup(2) × SpecialOrthogonalGroup(2)))
    return (:Euclid, :Euclid, :Circular)
end
function _manifoldtuple(
    ::typeof(TranslationGroup(2) × SpecialOrthogonalGroup(2) × TranslationGroup(2)),
)
    return (:Euclid, :Euclid, :Circular, :Euclid, :Euclid)
end
function _manifoldtuple(::typeof(TranslationGroup(3) × SpecialOrthogonalGroup(3)))
    return (:Euclid, :Euclid, :Euclid, :Circular, :Circular, :Circular)
end
function _manifoldtuple(
    ::typeof(SpecialOrthogonalGroup(3) × TranslationGroup(3) × TranslationGroup(3)),
)
    return (
        :Circular,
        :Circular,
        :Circular,
        :Euclid,
        :Euclid,
        :Euclid,
        :Euclid,
        :Euclid,
        :Euclid,
    )
end

"""
    $(SIGNATURES)

Calculate the KDE bandwidths for each dimension independly, as per manifold of each.  Return vector of all dimension bandwidths.
"""
function getKDEManifoldBandwidths(
    pts::AbstractMatrix{<:Real},
    manif::T1,
) where {T1 <: Tuple}
    #
    ndims = size(pts, 1)
    bws = ones(ndims)

    for i = 1:ndims
        if manif[i] == :Euclid
            bws[i] = getBW(kde!(pts[i, :]))[1, 1]
        elseif manif[i] == :Circular
            bws[i] = getBW(kde!_CircularNaiveCV(pts[i, :]))[1, 1]
        else
            error("Unrecognized manifold $(manif[i])")
        end
    end

    return bws
end

## ================================================================================================================================
# pass through API
## ================================================================================================================================

# not exported yet
# getManifold(x::ManifoldKernelDensity) = x.manifold

import KernelDensityEstimate: Ndim, Npts, getWeights, marginal
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld

Npts(::ManellicTree{M, D, N}) where {M, D, N} = N
Ndim(mt::ManellicTree) = manifold_dimension(mt.manifold)
getBW(mker::MvNormalKernel) = sqrt_Σ(mker) |> collect # cov(mker) |> collect
# getBW(::ManellicTree) currently only returns the permuted data as per .leaf_kernels
getBW(mt::ManellicTree) = getBW.(mt.leaf_kernels)

Ndim(x::ManifoldKernelDensity, w...; kw...) = Ndim(x.belief, w...; kw...)
Npts(x::ManifoldKernelDensity, w...; kw...) = Npts(x.belief, w...; kw...)

getWeights(x::ManifoldKernelDensity, w...; kw...) = getWeights(x.belief, w...; kw...)

getKDERange(x::ManifoldKernelDensity, w...; kw...) = getKDERange(x.belief, w...; kw...)
function getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...; kw...)
    return getKDERange((s -> s.belief).(x), w...; kw...)
end
getKDEMax(x::ManifoldKernelDensity, w...; kw...) = getKDEMax(x.belief, w...; kw...)
getKDEMean(x::ManifoldKernelDensity, w...; kw...) = getKDEMean(x.belief, w...; kw...)
getKDEfit(x::ManifoldKernelDensity, w...; kw...) = getKDEfit(x.belief, w...; kw...)

kld(x::ManifoldKernelDensity, w...; kw...) = kld(x.belief, w...; kw...)
minkld(x::ManifoldKernelDensity, w...; kw...) = minkld(x.belief, w...; kw...)

(x::ManifoldKernelDensity)(w...; kw...) = x.belief(w...; kw...)

#
