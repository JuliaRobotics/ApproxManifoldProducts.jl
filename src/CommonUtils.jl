# Common Utils


function resid2DLinear(μ, mus, Lambdas; diffop::Function=-)  # '-' exploits EuclideanManifold commutativity a-b = b-a
  dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
  # @show round.(dμ, digits=4)
  ret = sum( Lambdas.*dμ )
  return ret
end

function solveresid2DLinear!(res, x, mus, Lambdas; diffop::Function=-)::Nothing
  res[1] = resid2DLinear(x, mus, Lambdas, diffop=diffop)
  nothing
end

# import ApproxManifoldProducts: resid2DLinear, solveresid2DLinear
function solveresid2DLinear(res, x, mus, Lambdas; diffop::Function=-)::Float64
  solveresid2DLinear!(res, x, mus, Lambdas, diffop=diffop)
  return res[1]
end


"""
    $SIGNATURES

A clunky repeat calculation of one product kernel.
"""
function updateProductSample( dest::BallTreeDensity,
                              proposals::Vector{BallTreeDensity},
                              manifolds::Tuple,
                              smplIdx::Int,
                              labels::Vector{Int})
  #

  Ndens = length(proposals)
  Ndim = Ndim(dest)

  densLen = Npts.(proposals)

  calclambdas = zeros(Ndim, Ndens)
  calcmu = zeros(Ndim, Ndens)
  destMu = zeros(Ndim)
  destCov = 0.0

  @inbounds @fastmath @simd for dim in 1:Ndim
    for j in 1:Ndens
      calclambdas[dim,j] = 1.0/getBW(proposals[j])[dim,labels[j]]
      calcmu[dim,j] = getPoints(proposals[j])[dim,labels[j]]
    end
    destCov = getLambda(calclambdas)
    destCov = 1.0/destCov
    # μ = 1/Λ * Λμ  ## i.e. already scaled to mean only
    destMu[dim] = getMu(calcmu[dim, :], calclambdas[dim, :], destCov)
  end

  # previous points
  pts = getPoints(dest)
  pts[:,smplIdx] = destMu

  manikde!(pts, manifolds)
end


# TODO this should be a public method relating to getManifold
function _getManifoldFullOrPart(mkd::ManifoldKernelDensity, aspartial::Bool=true)
  if aspartial && isPartial(mkd)
    getManifoldPartial(mkd.manifold, mkd._partial)
  else
    mkd.manifold
  end
end

function Statistics.mean(mkd::ManifoldKernelDensity, aspartial::Bool=true; kwargs...)
  return mean(_getManifoldFullOrPart(mkd,aspartial), getPoints(mkd, aspartial); kwargs...)
end
"""
    $SIGNATURES

Alias for overloaded `Statistics.mean`.
"""
calcMean(mkd::ManifoldKernelDensity, aspartial::Bool=true) = mean(mkd, aspartial)

function Statistics.std(mkd::ManifoldKernelDensity, aspartial::Bool=true; kwargs...)
  std(_getManifoldFullOrPart(mkd,aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.var(mkd::ManifoldKernelDensity, aspartial::Bool=true; kwargs...)
  var(_getManifoldFullOrPart(mkd,aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.cov(mkd::ManifoldKernelDensity, aspartial::Bool=true; basis::Manifolds.AbstractBasis = Manifolds.DefaultOrthogonalBasis(), kwargs...)
  return cov(_getManifoldFullOrPart(mkd,aspartial), getPoints(mkd, aspartial); basis, kwargs... )
end
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
"""
function calcProductGaussians(M::AbstractManifold, 
                              μ_::Union{<:AbstractVector{P},<:NTuple{N,P}}, # point type commonly known as P 
                              Σ_::Union{Nothing,<:AbstractVector{S},<:NTuple{N,S}};
                                dim::Integer=manifold_dimension(M),
                              Λ_ = inv.(Σ_),
                              ) where {N,P,S<:AbstractMatrix{<:Real}}
  #
  # calc sum of covariances  
  Λ = zeros(MMatrix{dim,dim})
  Λ .= sum(Λ_)
  
  # Tangent space reference around the evenly weighted mean of incoming points
  u0 = mean(M, μ_)

  # calc the covariance weighted delta means of incoming points and covariances
  ΛΔμ = zeros(MVector{dim})
  for (s,u) in zip(Λ_, μ_)
    # require vee as per Pennec, Caesar Ref [3.6]
    Δuvee = vee(M, u0, log(M, u0, u))
    ΛΔμ += s*Δuvee
  end

  # calculate the delta mean
  Δμ = Λ \ ΛΔμ

  # return new mean and covariance
  return exp(M, u0, hat(M, u0, Δμ)), inv(Λ) 
end

# additional support case where covariances are passed as diagonal-only vectors 
# still pass nothing, to avoid stack overflow.  Only Λ_ is needed further
calcProductGaussians( M::AbstractManifold, 
                      μ_::Union{<:AbstractVector{P},<:NTuple{N,P}},
                      Σ_::Union{<:AbstractVector{S},<:NTuple{N,S}};
                        dim::Integer=manifold_dimension(M),
                      Λ_ = map(s->diagm( 1.0 ./ s), Σ_),
                      ) where {N,P,S<:AbstractVector} = calcProductGaussians(M, μ_, nothing; dim=dim, Λ_=Λ_ )
#

calcProductGaussians( M::AbstractManifold, 
                      μ_::Union{<:AbstractVector{P},<:NTuple{N,P}};
                        dim::Integer=manifold_dimension(M),
                      Λ_ = diagm.( (1.0 ./ μ_) ),
                      ) where {N,P} = calcProductGaussians(M, μ_, nothing; dim=dim, Λ_=Λ_ )
#


function _update!(dst::MN, src::MN) where {MN <: ManifoldKernelDensity}
  KDE._update!(dst.belief, src.belief)
  @assert dst._partial == src._partial "AMP._update! can only be done for exactly the same ._partial values in dst and src"
  setPointsMani!(dst._u0, src._u0)
  dst.infoPerCoord .= src.infoPerCoord
  
  dst
end


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




#
