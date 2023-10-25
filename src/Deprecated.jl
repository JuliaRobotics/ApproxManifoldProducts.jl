
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
@deprecate R(;x::Real=0.0,y::Real=0.0,z::Real=0.0) (M=SpecialOrthogonal(3);exp(M,identity_element(M),hat(M,Identity(M),[x,y,z]))) # convert(SO3, so3([x,y,z]))

export calcCovarianceBasic
# Returns the covariance (square), not deviation
function calcCovarianceBasic(M::AbstractManifold, ptsArr::Vector{P}) where P
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

@deprecate setPointsManiPartial!( Mdest::AbstractManifold, 
                                  dest, 
                                  Msrc::AbstractManifold, 
                                  src, 
                                  partial::AbstractVector{<:Integer},
                                  asPartial::Bool=true ) setPointPartial!( Mdest, dest, Msrc, src, partial, asPartial )


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
function productbelief( denspts::AbstractVector{P},
                        manifold::MB.AbstractManifold,
                        dens::Vector{<:ManifoldKernelDensity},
                        N::Int;
                        asPartial::Bool=false,
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P
  #

  @warn "productbelief is being deprecated, use manifoldProduct together with getPoints instead."
  mkd = AMP.manifoldProduct(dens, manifold, Niter=1, oldPoints=denspts, logger=logger)
  pGM = getPoints(mkd, asPartial)

  return pGM
end


# function calcMean(mkd::ManifoldKernelDensity{M}) where {M <: ManifoldsBase.AbstractManifold}
#   data = getPoints(mkd)
#   # Returns the mean point on manifold for consitency
#   mean(mkd.manifold, data)  
# end


@deprecate calcVariableCovarianceBasic(M::AbstractManifold, vecP::AbstractVector{P}) where P calcCovarianceBasic(M, vecP)


#