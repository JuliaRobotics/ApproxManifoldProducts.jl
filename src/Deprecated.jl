

## ======================================================================================================
## Remove below before v0.13
## ======================================================================================================


# function TUs.skew(v::SVector{3, T}) where {T <: Real}
#     # coordinates are the co-tangent elements
#     x, y, z = v[1], v[2], v[3]
#     # sum with default basis to form a tangent vector in the algebra
#     return SMatrix{3, 3, T}(+0, z, -y, -z, 0, x, +y, -x, 0)
# end

# Circular Manifold definition and associated arithmetic

# function get2DMu(
#     mus,
#     Lambdas;
#     diffop::Function = -,
#     periodicmanifold::Function = (x) -> x,
#     initrange::Tuple{Float64, Float64} = (-1e-5, 1e-5),
# )::Float64
#     # TODO: can be solved as the null space basis, but requires proper scaling
#     gg = (res, x) -> solveresid2DLinear!(res, x, mus, Lambdas; diffop = diffop)
#     initr = initrange[2] - initrange[1]
#     x0 = [initr * rand() + initrange[1]]
#     r = NLsolve.nlsolve(gg, x0)
#     xs = sign(r.zero[1] - x0[1]) * 1e-3 .+ r.zero
#     r = NLsolve.nlsolve(gg, xs)
#     return periodicmanifold(r.zero[1])
# end

# function get2DMuMin(
#     mus,
#     Lambdas;
#     diffop::Function = -,
#     periodicmanifold::Function = (x) -> x,
#     initrange::Tuple{Float64, Float64} = (-1e-5, 1e-5),
#     method = Optim.Newton(),
#     Λμ::Bool = false,
# )::Float64
#     # TODO: can be solved as the null space basis, but requires proper scaling
#     res = zeros(1)
#     # @show round.(mus, digits=3)
#     gg = (x) -> (solveresid2DLinear(res, x, mus, Lambdas; diffop = diffop))^2
#     initr = initrange[2] - initrange[1]
#     # TODO -- do we need periodicmanifold here?
#     x0 = [initr * rand() + initrange[1]]
#     r = Optim.optimize(gg, x0, method)
#     # @show r.minimizer[1]
#     return periodicmanifold(r.minimizer[1])
# end

# #


# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

# import TransformUtils.logmap_SO2

# export difftheta,
#     addtheta,
#     rbfAccAt!,
#     rbf!,
#     rbf,
#     evaluateManifoldNaive1D!,
#     manifoldLooCrossValidation,
#     kde!_CircularNaiveCV,
#     getCircMu,
#     getCircLambda

# const global reci_s2pi = 1.0 / sqrt(2.0 * pi) # 1.0/2.5066282746310002

# # On-manifold circular product callbacks

# # sign migth be flipped, but kept for legacy support -- FIXME this should be standardized!!
# difftheta(wth1, wth2) = log(_AMP_CIRCLE, wth2, wth1) # logmap_SO2(TUs.R(wth1)'*TUs.R(wth2))
# addtheta(wth1, wth2) = exp(_AMP_CIRCLE, wth2, wth1)   # TUs.wrapRad( wth1+wth2 )

# # manifold get Gaussian products mean
# function getCircMu(m::Vector{Float64}, s::Vector{Float64}, ::Float64)
#     return addtheta(0, get2DMu(m, s; diffop = difftheta, initrange = (-pi + 0.0, pi + 0.0)))
# end

# # getCircMu = (m::Vector{Float64}, s::Vector{Float64}, dummy::Float64) -> TUs.wrapRad(get2DMuMin(m, s, diffop=difftheta, initrange=(-pi+0.0,pi+0.0)))

# getCircLambda(x) = getEuclidLambda(x)

# """
#     $SIGNATURES

# Probability density function `p(x)`, as estimated by kernels
# ```math
# hatp_{-j}(x) = 1/(N-1) Σ_{i != j}^N frac{1}{sqrt{2pi}σ } exp{ -frac{(x-μ)^2}{2 σ^2} }
# ```
# """
# function normDistAccAt!(
#     ret::AV,
#     idx::Int,
#     x::Float64,
#     sigma::Float64,
#     w::Float64 = 1.0,
# ) where {AV <: AbstractVector}
#     global reci_s2pi
#     @fastmath ret[idx] += w * reci_s2pi / sigma * exp(-(x^2) / (2.0 * (sigma^2)))
#     return nothing
# end

# function rbfAccAt!(
#     ret::AV,
#     idx::Int,
#     x::Float64,
#     μ::Float64 = 0.0,
#     σ::Float64 = 1.0,
#     w::Float64 = 1.0,
#     diffop::Function = -,
# ) where {AV <: AbstractVector}
#     #
#     normDistAccAt!(ret, idx, diffop(x, μ), σ, w)
#     return nothing
# end
# function rbf!(
#     ret::AV,
#     x::Float64,
#     μ::Float64 = 0.0,
#     σ::Float64 = 1.0,
#     diffop::Function = -,
# ) where {AV <: AbstractVector}
#     #
#     ret[1] = 0.0
#     normDistAccAt!(ret, 1, diffop(x, μ), σ)
#     return nothing
# end

# function rbf(x::Float64, μ::Float64 = 0.0, σ::Float64 = 1.0)
#     ret = Vector{Float64}(undef, 1) # initialized in rbf!(..)
#     rbf!(ret, x, μ, σ)
#     return ret[1]
# end

# """
#     $SIGNATURES

# Evalute the KDE naively as equally weighted Gaussian kernels with common bandwidth.
# This function does, however, allow on-manifold evaluations.
# """
# function evaluateManifoldNaive1D!(
#     ret::Vector{Float64},
#     idx::Int,
#     pts::Array{Float64, 1},
#     bw::Float64,
#     x::Array{Float64, 1},
#     loo::Int = -1,
#     diffop = -,
# )
#     #
#     dontskip = loo == -1
#     N = length(pts)
#     reci_N = dontskip ? 1.0 / N : 1.0 / (N - 1)
#     for j = 1:N
#         if dontskip || loo != j
#             manifolddist = diffop(pts[loo], pts[j])
#             normDistAccAt!(ret, idx, manifolddist, bw, reci_N)
#         end
#     end

#     return nothing
# end
# function evaluateManifoldNaive1D!(
#     ret::Vector{Float64},
#     idx::Int,
#     bd::BallTreeDensity,
#     x::Array{Float64, 1},
#     loo::Int = -1,
#     diffop = -,
# )
#     #
#     return evaluateManifoldNaive1D!(
#         ret,
#         idx,
#         getPoints(bd)[:],
#         getBW(bd)[1, 1],
#         x,
#         loo,
#         diffop,
#     )
# end

# """
#     $SIGNATURES

# Calculate negative entropy with leave one out (j'th element) cross validation.

# Background
# ==========

# From: Silverman, B.: Density Estimation for Statistics and Data Analysis, 1986, p.52

# Probability density function `p(x)`, as estimated by kernels
# ```math
# hatp_{-j}(x) = 1/(N-1) Σ_{i != j}^N frac{1}{sqrt{2pi}σ } exp{ -frac{(x-μ)^2}{2 σ^2} }
# ```
# and has Cross Validation number as the average log evaluations of leave one out `hatp_{-j}(x)`:
# ```math
# CV(p) = 1/N Σ_i^N log hat{p}_{-j}(x_i)
# ```

# This quantity `CV` is related to an entropy `H(p)` estimate via:
# ```math
# H(p) = -CV(p)
# ```
# """
# function manifoldLooCrossValidation(
#     pts::Array,
#     bw::Float64;
#     own::Bool = true,
#     diffop::Function = -,
# )
#     #
#     N = maximum(size(pts))
#     h = [bw;]
#     loo = zeros(N)
#     @inbounds for i = 1:N
#         if !own
#             # validation version
#             pts99 = pts[[1:(i - 1); (i + 1):end]]
#             p99 = kde!(pts99, h)
#             loo[i] = log(p99([pts[i];])[1])
#         else
#             # own naive entropy calculation
#             loo[i] = 0.0
#             evaluateManifoldNaive1D!(loo, i, pts, bw, pts, i, diffop)
#             loo[i] = log(loo[i])
#         end
#     end
#     return sum(loo) / N
# end

# function kde!_CircularNaiveCV(points::AbstractVector)
#     # initial setup parameters
#     dims = 1 # size(points,1)
#     bwds = zeros(dims)
#     # initial testing values
#     lower = 0.001
#     upper = 2pi

#     # excessive for loop for leave one out likelihiood cross validation (Silverman 1986, p.52)
#     for i = 1:dims
#         minEntropyLOOCV =
#             (bw) -> -manifoldLooCrossValidation(points, bw; own = true, diffop = difftheta)
#         res = Optim.optimize(
#             minEntropyLOOCV,
#             lower,
#             upper,
#             Optim.GoldenSection();
#             x_tol = 0.001,
#         )
#         bwds[i] = res.minimizer
#     end

#     # cosntruct the kde with CV optimized bandwidth
#     p = kde!(points, bwds, (addtheta,), (difftheta,))

#     return p
# end

# #


# Euclidean Manifold definitions and arithmetic

# get2DLambda(Lambdas::AbstractVector{<:Real}) = sum(Lambdas)

# #


# legacy content to facilitate transition to AMP


# function resid2DLinear(μ, mus, Lambdas; diffop::Function = -)  # '-' exploits EuclideanManifold commutativity a-b = b-a
#     # dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
#     # @show round.(dμ, digits=4)
#     # ret = sum( Lambdas.*dμ )
#     r = map((mu, lam) -> diffop(μ[], mu) * lam, mus, Lambdas)
#     return sum(r)
# end

# function solveresid2DLinear!(res, x, mus, Lambdas; diffop::Function = -)::Nothing
#     res[1] = resid2DLinear(x, mus, Lambdas; diffop = diffop)
#     return nothing
# end

# # import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
# function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function = -)::Float64
#     solveresid2DLinear!(res, x, mus, Lambdas; diffop = diffop)
#     return res[1]
# end


# function _update!(dst::MN, src::MN) where {MN <: ManifoldKernelDensity}
#     KDE._update!(dst.belief, src.belief)
#     @assert dst._partial == src._partial "AMP._update! can only be done for exactly the same ._partial values in dst and src"
#     setPointsMani!(dst._u0, src._u0)
#     dst.infoPerCoord .= src.infoPerCoord

#     return dst
# end


# function _reducePartialManifoldElements(el::Symbol)
#     if el == :Euclid
#         return TranslationGroup(1)
#     elseif el == :Circular
#         return Circle()
#     end
#     return error("unknown manifold_symbol $el")
# end

# """
#     $SIGNATURES

# Lots to do here, see RoME.jl #244 and standardized usage with Manifolds.jl.

# Notes
# - diffop( test, reference )   <===>   ΔX = inverse(test) * reference

# DevNotes
# - FIXME replace with Manifolds.jl #41, RoME.jl #244
# """
# function buildHybridManifoldCallbacks(manif::Tuple)
#     # TODO use multiple dispatch instead -- will be done for second version of system
#     addopT = []
#     diffopT = []
#     getManiMu = []
#     getManiLam = []

#     for mn in manif
#         if mn == :Euclid
#             push!(addopT, +)
#             push!(diffopT, -)
#             push!(getManiMu, KDE.getEuclidMu)
#             push!(getManiLam, KDE.getEuclidLambda)
#         elseif mn == :Circular
#             push!(addopT, addtheta)
#             push!(diffopT, difftheta)
#             push!(getManiMu, getCircMu)
#             push!(getManiLam, getCircLambda)
#         else
#             error("Unrecognized manifold $(mn)")
#         end
#     end

#     return (addopT...,), (diffopT...,), (getManiMu...,), (getManiLam...,)
# end

# # FIXME TO BE REMOVED
# # _MtoSymbol(::Euclidean{Tuple{1}}) = :Euclid
# # _MtoSymbol(::Circle) = :Circular

# function _manifoldtuple(M::AbstractManifold)
#     # TODO WIP to remove convert(Tuple, M) type piracy.
#     # Also easier dev experience without so many convert (800+) and implicit conversion.
#     @warn "Please use `_manifoldtuple(M)` instead. This will be removed (hopefully soon). Got" typeof(
#         M,
#     )
#     return convert(Tuple, M)
# end
# # _manifoldtuple(M::ProductManifold) = _MtoSymbol.(M.manifolds)
# # _manifoldtuple(M::Manifolds.TranslationGroup) = tuple([:Euclid for i in 1:manifold_dimension(M)]...)
# function _manifoldtuple(M::LieGroups.TranslationGroup)
#     return tuple([:Euclid for i = 1:manifold_dimension(M)]...)
# end
# _manifoldtuple(::typeof(LieGroups.CircleGroup(ℝ))) = (:Circular,)
# function _manifoldtuple(
#     ::LieGroup{ℂ, AbelianMultiplicationGroupOperation, Manifolds.Circle{ℂ}},
# )
#     return (:Euclid,)
# end
# _manifoldtuple(M::ValidationLieGroup) = _manifoldtuple(M.lie_group)

# function _manifoldtuple(::Manifolds.Euclidean{Tuple{N}, ℝ}) where {N}
#     return tuple([:Euclid for i = 1:N]...)
# end
# # _manifoldtuple(::Manifolds.Circle{ℝ})  = error("#FIXME")#(:Circular,)
# # _manifoldtuple(::Manifolds.RealCircleGroup)  = (:Circular,)

# _manifoldtuple(::typeof(Euclid)) = (:Euclid,)
# _manifoldtuple(::typeof(Euclid2)) = (:Euclid, :Euclid)
# _manifoldtuple(::typeof(Euclid3)) = (:Euclid, :Euclid, :Euclid)
# _manifoldtuple(::typeof(Euclid4)) = (:Euclid, :Euclid, :Euclid, :Euclid)

# _manifoldtuple(::typeof(SpecialOrthogonalGroup(2))) = (:Circular,)
# _manifoldtuple(::typeof(SpecialOrthogonalGroup(3))) = (:Circular, :Circular, :Circular)
# function _manifoldtuple(::typeof(SpecialEuclideanGroup(2; variant = :right)))
#     return (:Euclid, :Euclid, :Circular)
# end
# function _manifoldtuple(::typeof(SpecialEuclideanGroup(3; variant = :right)))
#     return (:Euclid, :Euclid, :Euclid, :Circular, :Circular, :Circular)
# end
# function _manifoldtuple(::typeof(TranslationGroup(2) × SpecialOrthogonalGroup(2)))
#     return (:Euclid, :Euclid, :Circular)
# end
# function _manifoldtuple(
#     ::typeof(TranslationGroup(2) × SpecialOrthogonalGroup(2) × TranslationGroup(2)),
# )
#     return (:Euclid, :Euclid, :Circular, :Euclid, :Euclid)
# end
# function _manifoldtuple(::typeof(TranslationGroup(3) × SpecialOrthogonalGroup(3)))
#     return (:Euclid, :Euclid, :Euclid, :Circular, :Circular, :Circular)
# end
# function _manifoldtuple(
#     ::typeof(SpecialOrthogonalGroup(3) × TranslationGroup(3) × TranslationGroup(3)),
# )
#     return (
#         :Circular,
#         :Circular,
#         :Circular,
#         :Euclid,
#         :Euclid,
#         :Euclid,
#         :Euclid,
#         :Euclid,
#         :Euclid,
#     )
# end

# """
#     $(SIGNATURES)

# Calculate the KDE bandwidths for each dimension independly, as per manifold of each.  Return vector of all dimension bandwidths.
# """
# function getKDEManifoldBandwidths(
#     pts::AbstractMatrix{<:Real},
#     manif::T1,
# ) where {T1 <: Tuple}
#     #
#     ndims = size(pts, 1)
#     bws = ones(ndims)

#     for i = 1:ndims
#         if manif[i] == :Euclid
#             bws[i] = getBW(kde!(pts[i, :]))[1, 1]
#         elseif manif[i] == :Circular
#             bws[i] = getBW(kde!_CircularNaiveCV(pts[i, :]))[1, 1]
#         else
#             error("Unrecognized manifold $(manif[i])")
#         end
#     end

#     return bws
# end

# ## ================================================================================================================================
# # pass through API
# ## ================================================================================================================================

# # not exported yet
# # getManifold(x::ManifoldKernelDensity) = x.manifold

# import KernelDensityEstimate: Ndim, Npts, getWeights, marginal
# import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
# import KernelDensityEstimate: sample, rand, resample, kld, minkld

# Npts(::ManellicTree{M, D, N}) where {M, D, N} = N
# Ndim(mt::ManellicTree) = manifold_dimension(mt.manifold)
# getBW(mker::MvNormalKernel) = sqrt_Σ(mker) |> collect # cov(mker) |> collect
# # getBW(::ManellicTree) currently only returns the permuted data as per .leaf_kernels
# getBW(mt::ManellicTree) = getBW.(mt.leaf_kernels)

# Ndim(x::ManifoldKernelDensity, w...; kw...) = Ndim(x.belief, w...; kw...)
# Npts(x::ManifoldKernelDensity, w...; kw...) = Npts(x.belief, w...; kw...)

# getWeights(x::ManifoldKernelDensity, w...; kw...) = getWeights(x.belief, w...; kw...)

# getKDERange(x::ManifoldKernelDensity, w...; kw...) = getKDERange(x.belief, w...; kw...)
# function getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...; kw...)
#     return getKDERange((s -> s.belief).(x), w...; kw...)
# end
# getKDEMax(x::ManifoldKernelDensity, w...; kw...) = getKDEMax(x.belief, w...; kw...)
# getKDEMean(x::ManifoldKernelDensity, w...; kw...) = getKDEMean(x.belief, w...; kw...)
# getKDEfit(x::ManifoldKernelDensity, w...; kw...) = getKDEfit(x.belief, w...; kw...)

# kld(x::ManifoldKernelDensity, w...; kw...) = kld(x.belief, w...; kw...)
# minkld(x::ManifoldKernelDensity, w...; kw...) = minkld(x.belief, w...; kw...)

# (x::ManifoldKernelDensity)(w...; kw...) = x.belief(w...; kw...)

# #

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

##