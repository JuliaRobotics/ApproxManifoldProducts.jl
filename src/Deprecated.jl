


# function ManifoldKernelDensity(
#     M::MB.AbstractManifold,
#     vecP::AbstractVector{P},
#     u0 = vecP[1]; # vecP[1]
#     partial::L = nothing,
#     partl_cb::Union{Nothing, <:Function} = nothing,
#     infoPerCoord::AbstractVector{<:Real} = ones(getNumberCoords(M, u0)),
#     dims::Int = manifold_dimension(M),
#     bw::Union{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}, Nothing} = nothing,
#     belmodel::Function = (a, b, aF, dF) ->
#         KernelDensityEstimate.kde!(a, collect(b), aF, dF), # collect(b) but error length(::Nothing)
# ) where {P, L}
#     #
#     # FIXME obsolete
#     arr = Matrix{Float64}(undef, dims, length(vecP))

#     for j = 1:length(vecP)
#         arr[:, j] = makeCoordsFromPoint(M, vecP[j])
#     end

#     # FIXME ON FIRE REMOVE LEGACY
#     manis = _manifoldtuple(M)
#     # find or have the bandwidth
#     _bw = isnothing(bw) ? getKDEManifoldBandwidths(arr, manis) : bw
#     # NOTE workaround for partials and user did not specify a bw
#     if isnothing(bw) && !isnothing(partial)
#         mask = ones(Int, length(_bw)) .== 1
#         mask[partial] .= false
#         _bw[mask] .= 1.0
#     end
#     # FIXME ON FIRE REMOVE LEGACY
#     addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
#     bel = belmodel(arr, _bw, addopT, diffopT)
#     # bel = KernelDensityEstimate.kde!(arr, collect(_bw), addopT, diffopT)
#     return ManifoldKernelDensity(M, bel, partial, u0, infoPerCoord)
# end

# internal workaround function for building partial submanifold dimensions, must be upgraded/standarized
# function _buildManifoldPartial(fullM::MB.AbstractManifold, partial_coord_dims)
#     #
#     # temporary workaround during Manifolds.jl integration
#     manif = _manifoldtuple(fullM)[partial_coord_dims]
#     # 
#     newMani = MB.AbstractManifold[]
#     for me in manif
#         push!(newMani, _reducePartialManifoldElements(me))
#     end

#     # assume independent dimensions for definition, ONLY USED AS DECORATOR AT THIS TIME, FIXME
#     return ProductManifold(newMani...)
# end

# function Statistics.mean(mkd::ManifoldKernelDensity; kwargs...)
#   return mean(mkd.manifold, getPoints(mkd); kwargs...)
# end
# function Statistics.cov(mkd::ManifoldKernelDensity; kwargs...) 
#   cov(mkd.manifold, getPoints(mkd); kwargs...)
# end
# function Statistics.std(mkd::ManifoldKernelDensity; kwargs...)
#   return std(mkd.manifold, getPoints(mkd); kwargs...)
# end
# function Statistics.var(mkd::ManifoldKernelDensity; kwargs...)
#   return var(mkd.manifold, getPoints(mkd); kwargs...)
# end

# function Base.convert(
#     ::Type{B},
#     mkd::ManifoldKernelDensity{M, B},
# ) where {M, B <: BallTreeDensity}
#     return mkd.belief
# end

# """
# Likely new bug see KDE #70 
# """
# function _buildDensityProductElements(
#     XX::AbstractVector{B};
#     outName::Symbol = :product,
#     inNames::Union{<:AbstractVector{Symbol}, NTuple{D, Symbol}} = [
#         Symbol("belief$i") for i = 1:length(XX)
#     ],
#     inFctNames::Union{<:AbstractVector{Symbol}, NTuple{D, Symbol}} = [
#         Symbol("factor$i") for i = 1:length(XX)
#     ],
#     _glbs = KDE.makeEmptyGbGlb(; recordChoosen = true),
#     product::B = *(XX; glbs = _glbs, addEntropy = false),
# ) where {B <: BallTreeDensity, D}
#     #

#     npts = Npts(product)
#     ndim = Ndim(product)
#     ndens = length(XX)

#     # pts = getPoints(product)
#     # @cast outArr[j][i] := pts[i,j]
#     outArr = [getPoints(product, i) for i = 1:npts]
#     bw = [getBW(product)[:, i] for i = 1:npts]
#     lblComb = [zeros(Int, ndens) for i = 1:npts]

#     # restructure points for incoming densities
#     # indens = (XX...,)
#     XXarr = [([getPoints(x, i) for i = 1:Npts(x)]) for x in XX]
#     BWarr = [([getBW(x)[:, i] for i = 1:Npts(x)]) for x in XX]

#     # build the object
#     mdp = DensityProductElements(
#         outArr,
#         bw,
#         outName,
#         Ref(true),
#         lblComb,
#         (XXarr...,),
#         (BWarr...,),
#         (inNames...,),
#         (inFctNames...,),
#         Channel{Pair{Int, Int}}(100),
#     )

#     # set the labels selections used for current product
#     if length(XX) == 1
#         resize!(mdp.lblCombinations, npts)
#         for i = 1:npts
#             resize!(mdp.lblCombinations[i], 1)
#             mdp.lblCombinations[i][1] = i
#         end
#     else
#         _setLabelCombinations!(mdp, _glbs.labelsChoosen)
#     end

#     return mdp
# end


# function getPoints(
#     x::ManifoldKernelDensity{M, B, L},
#     aspartial::Bool = true;
#     permute::Bool = true,
# ) where {M <: AbstractManifold, B <: BallTreeDensity, L <: AbstractVector{Int}}
#     #
#     pts = getPoints(x.belief, permute)

#     (M_, pts_, u0_) = if (L !== nothing) && aspartial
#         Mp, Rp, lkup = getManifoldPartial(x.manifold, x._partial, x._u0)
#         (Mp, view(pts, x._partial, :), Rp)
#     else
#         (x.manifold, pts, x._u0)
#     end

#     return _matrixCoordsToPoints(M_, pts_, u0_)
# end

# # can only do for Array, not view
# function _setProductElements!(mdp::DensityProductElements{D}, 
#                               prd::BallTreeDensity)
#   #
#   # also set the bandwidth
#   dim = Ndim(prd)
#   resize!(mdp.outBW, dim)

#   npts = Npts(prd)
#   for i in 1:
#     mdp.outBW[:,i] .= getBW(prd)[:,1] # fix for all elements
#   end

#   # set kernel center elements
#   resize!(mdp.outElements, )
#   for i in 1:Npts(prd)
#     resize!(mdp.outElements[i], dim)
#     mdp.outElements[i][:] .= getPoints(prd, i)
#   end

#   #
#   nothing
# end

# TODO is this function obsolete?
# function getManifoldPartial(
#     M::TranslationGroup{Tuple{N}},
#     partial::AbstractVector{Int},
#     repr::_PartiableRepresentationFlat{T} = nothing,
#     offset::Base.RefValue{Int} = Ref(0);
#     doError::Bool = true,
# ) where {N, T <: Number}
#     #
#     mask = _checkManifoldPartialDims(M, partial, offset, doError)
#     offset[] += manifold_dimension(M)
#     len = sum(mask)
#     repr_p = repr === nothing ? nothing : zeros(T, len)
#     return (TranslationGroup(len), repr_p)
# end

# function getManifoldPartial(M::AbstractLieGroup, 
#                             partial::AbstractVector{<:Integer}, 
#                             repr::_PartiableRepresentation=nothing,
#                             offset::Base.RefValue{<:Integer}=Ref(0);
#                             doError::Bool=true )
#   #
#   # mask the desired coordinate dimensions
#   mask = _checkManifoldPartialDims(M,partial,offset, doError)

#   if sum(mask) == manifold_dimension(M)
#     # asking for all coordinate dimensions as offered by M
#     return (M,repr)
#   end
#   # recursion may need to branch for ProductManifold
#   # Note loss of the Group operation information at this time
#   getManifoldPartial(M.manifold, partial, repr, offset, doError=doError)
# end

# # TODO, deprecate convert approach and using constructor helpers instead
# function convert(
#     ::Type{MvNormalKernel{
#         ApproxManifoldProducts.DensityKernel{
#             L,
#             MvNormal{F,P,Z},
#             S
#         }
#     }},
#     src::MvNormalKernel,
# ) where {L,F,P,Z,S}

#     _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
#     _sap(::Type{ArrayPartition{T,_S}}) where {T,_S} = _S
#     _new(s) = S(s)
#     _new(s::ArrayPartition{T,O}) where {T,O} = ArrayPartition(begin
#         S_ = _sap(S)
#         [S_.parameters[i](v) for (i,v) in enumerate(s.x)]
#     end...)

#     m = _new(src.shim.params)

#     MvNormalKernel(
#         m,
#         _matType(P)(cov(src.shim.functional)),
#         src.shim.weight;
#         partial = L,
#         # partl_cb,
#     )
# end


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

# function marginal(
#     x::ManifoldKernelDensity{M, B, L},
#     dims::AbstractVector{<:Integer},
# ) where {M <: AbstractManifold, B, L <: AbstractVector{<:Integer}}
#     #
#     ldims::Vector{Int} = intersect(x._partial, dims)
#     return ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
# end
# # manis = convert(Tuple, x.manifold)
# # partMani = _reducePartialManifoldElements(manis[dims])
# # pts = getPoints(x)

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
