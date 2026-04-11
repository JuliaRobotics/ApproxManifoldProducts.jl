

# @kwdef struct MvNormalKernel{P, T, M, iM} <: AbstractKernel
#     """ On-manifold point representing center (mean) of the MvNormal distribution """
#     μ::P
#     """ Zero-mean normal distribution with covariance """
#     p::MvNormal{T, M}
#     # TDB might already be covered in p.Σ.chol but having issues with SymPD (not particular to this AMP repo)
#     """ Manually maintained square root concentration matrix for faster compute, TODO likely duplicate of existing Distrubtions.jl functionality. """
#     sqrt_iΣ::iM = sqrt(inv(cov(p)))
#     """ Nonparametric weight value """
#     weight::Float64 = 1.0
# end


# # deprecated as legacy and replaced by previously called manikde!_manellic
# function manikde!(
#     M::MB.AbstractManifold,
#     vecP::AbstractVector{P},
#     u0::P = vecP[1];
#     kw...,
# ) where {P}
#     return ManifoldKernelDensity(M, vecP, u0; kw...)
# end


# """
#     $SIGNATURES

# Once a Gibbs product is available, this function can be used to update the product assuming some change to the input
# to some or some or all of the input density kernels.

# Notes
# - This function does not resample a new posterior sample pairing of inputs, only updates with existing 
# """
# function _updateMetricTreeDensityProduct( npd0::BallTreeDensity,
#                                           trees::Array{BallTreeDensity,1},
#                                           anFcns,
#                                           anParams;
#                                           Niter::Int=3,
#                                           addop::Tuple=(+,),
#                                           diffop::Tuple=(-,),
#                                           getMu::Tuple=(getEuclidMu,),
#                                           getLambda::T4=(getEuclidLambda,),
#                                           glbs = makeEmptyGbGlb(),
#                                           addEntropy::Bool=true )
#   #

# end



# # default replace non-partial/non-marginal values
# # Trivial case where no information from destination is kept, only from src.
# function Base.replace(
#     ::ManifoldKernelDensity{M, B, Nothing},
#     src::ManifoldKernelDensity{M, B, Nothing},
# ) where {M <: AbstractManifold, B}
#     #
#     return src
# end

# # replace dest non-partial with incoming partial values
# function Base.replace(
#     dest::ManifoldKernelDensity{M, B, Nothing},
#     src::ManifoldKernelDensity{M, B, <:AbstractVector},
# ) where {M <: AbstractManifold, B}
#     #
#     pl = src._partial
#     # FIXME what about 
#     destPts = getPoints(dest.belief)
#     # get source partial points only 
#     newPts = getPoints(src.belief)
#     @assert size(destPts, 2) <= size(newPts, 2) "MKD replace currently requires the number of points to be the same, dest=$(size(destPts,2)), src=$(size(newPts,2))"
#     # TODO use eachindex or axes instead of 1:size
#     for i = 1:size(destPts, 2)
#         destPts[pl, i] .= newPts[pl, i]
#     end
#     # and new bandwidth
#     oldBw = getBW(dest.belief)[:, 1]
#     oldBw[pl] .= getBW(src.belief)[pl, 1]

#     # finaly update the belief with a new container
#     newBel = kde!(destPts, oldBw)

#     # also set the metadata values
#     ipc = deepcopy(dest.infoPerCoord)
#     ipc[pl] .= src.infoPerCoord[pl]

#     # and _u0 point is a bit more tricky
#     c0 = collect(vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, dest._u0)))
#     c_ = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, src._u0))
#     c0[pl] .= c_[pl]
#     u0 = exp(dest.manifold, dest._u0, hat(dest.manifold, dest._u0, c0))

#     # return the update destimation ManifoldKernelDensity object
#     return ManifoldKernelDensity(dest.manifold, newBel, nothing, u0; infoPerCoord = ipc)
# end

# # replace partial/marginal with different incoming partial values
# function Base.replace(
#     dest::ManifoldKernelDensity{M, B, <:AbstractVector},
#     src::ManifoldKernelDensity{M, B, <:AbstractVector},
# ) where {M <: AbstractManifold, B}
#     #
#     pl = src._partial
#     destPts = getPoints(dest.belief)
#     # get source partial points only 
#     newPts = getPoints(src.belief)
#     @assert size(newPts, 2) == size(destPts, 2) "this replace currently requires the number of points to be the same, dest=$(size(destPts,2)), src=$(size(newPts,2))"
#     for i = 1:size(destPts, 2)
#         destPts[pl, i] .= newPts[pl, i]
#     end
#     # and new bandwidth
#     oldBw = getBW(dest.belief)[:, 1]
#     oldBw[pl] .= getBW(src.belief)[pl, 1]

#     # finaly update the belief with a new container
#     newBel = kde!(destPts, oldBw)

#     # also set the metadata values
#     ipc = deepcopy(dest.infoPerCoord)
#     ipc[pl] .= src.infoPerCoord[pl]

#     # and _u0 point is a bit more tricky
#     c0 = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, dest._u0))
#     c_ = vee(dest.manifold, dest._u0, log(dest.manifold, dest._u0, src._u0))
#     c0[pl] .= c_[pl]
#     u0 = exp(dest.manifold, dest._u0, hat(dest.manifold, dest._u0, c0))

#     # and update the partial information
#     pl_ = union(dest._partial, pl)

#     # return the update destimation ManifoldKernelDensity object
#     if length(pl_) == manifold_dimension(dest.manifold)
#         # no longer a partial/marginal
#         return ManifoldKernelDensity(dest.manifold, newBel, nothing, u0; infoPerCoord = ipc)
#     else
#         # still a partial
#         return ManifoldKernelDensity(dest.manifold, newBel, pl_, u0; infoPerCoord = ipc)
#     end
# end


## ======================================================================================================
## Remove below before v0.10
## ======================================================================================================

# function setPointsMani!(dest::ProductRepr, src::ProductRepr)
#   for (k,prt) in enumerate(dest.parts)
#     setPointsMani!(prt, src.parts[k])
#   end
# end

## ======================================================================================================
## Remove below before v0.8
## ======================================================================================================

# function __init__()
#   @require Gadfly="c91e804a-d5a3-530f-b6f0-dfbca275c004" begin
#     @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("plotting/CircularPlotting.jl")
#   end
# end

@deprecate R(th::Real) _Rot.RotMatrix2(th).mat # = [[cos(th);-sin(th)]';[sin(th);cos(th)]'];
@deprecate R(; x::Real = 0.0, y::Real = 0.0, z::Real = 0.0) (
    M = SpecialOrthogonalGroup(3); exp(
        M,
        identity_element(M),
        hat(M, Identity(M), [x, y, z]),
    )
) # convert(SO3, so3([x,y,z]))

export calcCovarianceBasic
# Returns the covariance (square), not deviation
function calcCovarianceBasic(M::AbstractManifold, ptsArr::Vector{P}) where {P}
    @warn "`calcCovarianceBasic` is deprecated. Replace with IIF.calcSTDBasicSpread from IIF or `cov` or `var` from Manifolds. See issue AMP#150."
    μ = mean(M, ptsArr)
    Xcs = vee.(Ref(M), Ref(μ), log.(Ref(M), Ref(μ), ptsArr))
    Σ = mean(Xcs .* transpose.(Xcs))
    msst = Σ
    msst_ = 0 < sum(1e-10 .< msst) ? maximum(msst) : 1.0
    return msst_
end

## ======================================================================================================
## Remove below before v0.7
## ======================================================================================================

# export
#   coords,
#   uncoords,
#   getPointsManifold

## New Manifolds.jl aware API -- TODO find the right file placement

# # TODO, hack, use the proper Manifolds.jl intended vectoration methods instead
# _makeVectorManifold(::MB.AbstractManifold, arr::AbstractArray{<:Real}) = arr
# _makeVectorManifold(::MB.AbstractManifold, val::Real) = [val;]
# _makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(2))} = coords(M, prr)
# _makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(3))} = coords(M, prr)

## ======================================================================================================
## Remove below before v0.6
## ======================================================================================================

@deprecate setPointsManiPartial!(
    Mdest::AbstractManifold,
    dest,
    Msrc::AbstractManifold,
    src,
    partial::AbstractVector{<:Integer},
    asPartial::Bool = true,
) setPointPartial!(Mdest, dest, Msrc, src, partial, asPartial)

export productbelief

"""
    $SIGNATURES

Take product of `dens` (including optional partials beliefs) as proposals to be multiplied together.

Notes
-----
- Return points of full dimension, even if only partial dimensions in proposals.
  - 'Other' dimensions left unchanged from incoming `denspts`
- `d` dimensional product approximation
- Incorporate ApproxManifoldProducts to process variables in individual batches.

DevNotes
- TODO Consolidate with [`AMP.manifoldProduct`](@ref), especially concerning partials. 
"""
function productbelief(
    denspts::AbstractVector{P},
    manifold::MB.AbstractManifold,
    dens::Vector{<:ManifoldKernelDensity},
    N::Int;
    asPartial::Bool = false,
    dbg::Bool = false,
    logger = ConsoleLogger(),
) where {P}
    #

    @warn "productbelief is being deprecated, use manifoldProduct together with getPoints instead."
    mkd =
        AMP.manifoldProduct(dens, manifold; Niter = 1, oldPoints = denspts, logger = logger)
    pGM = getPoints(mkd, asPartial)

    return pGM
end

# function calcMean(mkd::ManifoldKernelDensity{M}) where {M <: ManifoldsBase.AbstractManifold}
#   data = getPoints(mkd)
#   # Returns the mean point on manifold for consitency
#   mean(mkd.manifold, data)  
# end

@deprecate calcVariableCovarianceBasic(
    M::AbstractManifold,
    vecP::AbstractVector{P},
) where {P} calcCovarianceBasic(M, vecP)

#
