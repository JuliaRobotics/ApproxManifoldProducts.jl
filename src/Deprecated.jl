
## ======================================================================================================
## Remove below before v0.8
## ======================================================================================================




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