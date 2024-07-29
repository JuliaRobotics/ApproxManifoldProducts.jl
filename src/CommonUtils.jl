# Common Utils


function resid2DLinear(μ, mus, Lambdas; diffop::Function=-)  # '-' exploits EuclideanManifold commutativity a-b = b-a
  # dμ = broadcast(diffop, μ, mus)  # mus .- μ  ## μ .\ mus
  # @show round.(dμ, digits=4)
  # ret = sum( Lambdas.*dμ )
  r = map((mu, lam) -> diffop(μ[], mu) * lam, mus, Lambdas)
  return sum(r)
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
  return mean(_getManifoldFullOrPart(mkd,aspartial), getPoints(mkd, aspartial), GeodesicInterpolation(); kwargs...)
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

_makevec(w::AbstractVector) = w
_makevec(w::Tuple) = [w...]


function calcProductGaussians_flat(
  M::AbstractManifold, 
  μ_::Union{<:AbstractVector{P},<:NTuple{N,P}}, # point type commonly known as P (actually on-manifold)
  Σ_::Union{<:AbstractVector{S},<:NTuple{N,S}};
  μ0 = mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
  Λ_ = inv.(Σ_),
  weight::Real = 1.0,
  do_transport_correction::Bool = true
) where {N,P<:AbstractArray,S<:AbstractMatrix{<:Real}}
  # calc sum of covariances  
  Λ = +(Λ_...)
  
  # calc the covariance weighted delta means of incoming points and covariances
  ΛΔμc = mapreduce(+, zip(Λ_, μ_)) do (s,u)
    Δuvee = vee(M, μ0, log(M, μ0, u))
    s*Δuvee
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
  μ_::Union{<:AbstractVector{P},<:NTuple{N,P}}, # point type commonly known as P (actually on-manifold)
  Σ_::Union{<:AbstractVector{S},<:NTuple{N,S}};
  μ0 = mean(M, _makevec(μ_)), # Tangent space reference around the evenly weighted mean of incoming points
  Λ_ = inv.(Σ_),    # TODO these probably need to be transported to common tangent space `μ0` -- FYI @Affie 24Q2
  weight::Real = 1.0,
  do_transport_correction::Bool = true
) where {N,P<:AbstractArray,S<:AbstractMatrix{<:Real}}
  # step 1, basic/naive Gaussian product (ignoring disjointed covariance coordinates) 
  Δμn, Σn = calcProductGaussians_flat(M, μ_, Σ_; μ0, Λ_, weight)
  Δμ = exp(M, μ0, hat(M, μ0, Δμn))

  # for development and testing cases return without doing transport
  do_transport_correction ? nothing : (return Δμ, Σn)
  
  # FIXME first transport (push forward) covariances to common coordinates
  # see [Ge, van Goor, Mahony, 2024]
  iΔμ = inv(M, Δμ)
  μi_ = map(u->Manifolds.compose(M,iΔμ,u), μ_)
  μi_̂  = map(u->log(M,μ0,u), μi_)
  # μi = map(u->vee(M,μ0,u), μi_̂ )
  Ji = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie.(Ref(M), μi_̂ )
  iJi = inv.(Ji)
  Σi_hat = map((J,S)->J*S*(J'), iJi, Σ_)
  
  # Reset step to absorb extended μ+ coordinates into kernel on-manifold μ 
  # consider using Δμ in place of μ0
  Δμplusc, Σdiam = ApproxManifoldProducts.calcProductGaussians_flat(M, μi_, Σi_hat; μ0, weight)
  Δμplus_̂  = hat(M, μ0, Δμplusc)
  Δμplus = exp(M, μ0, Δμplus_̂ )
  μ_plus = Manifolds.compose(M,Δμ,Δμplus)
  Jμ = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, Δμplus_̂ )
  # iJμ = inv(Jμ)
  Σ_plus = Jμ*Σdiam*(Jμ')
  
  # return new mean and covariance
  return μ_plus, Σ_plus 
end

# additional support case where covariances are passed as diagonal-only vectors 
# still pass nothing, to avoid stack overflow.  Only Λ_ is needed further
calcProductGaussians( 
  M::AbstractManifold, 
  μ_::Union{<:AbstractVector{P},<:NTuple{N,P}},
  Σ_::Union{<:AbstractVector{S},<:NTuple{N,S}};
  dim::Integer=manifold_dimension(M),
  Λ_ = map(s->diagm( 1.0 ./ s), Σ_),
  weight::Real = 1.0,
  do_transport_correction::Bool = true
) where {N,P,S<:AbstractVector} = calcProductGaussians(M, μ_, nothing; dim, Λ_, do_transport_correction )
#

# # FIXME, review `./μ_`, what is this?  nan risk?
# calcProductGaussians( 
#   M::AbstractManifold, 
#   μ_::Union{<:AbstractVector{P},<:NTuple{N,P}};
#   Λ_ = diagm.( (1.0 ./ μ_) ),
#   weight::Real = 1.0,
# ) where {N,P} = calcProductGaussians(M, μ_, nothing; Λ_ )


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
  kernels::Union{<:AbstractVector{K},NTuple{N,K}};
  μ0 = nothing,
  weight::Real = 1.0,
  do_transport_correction::Bool = true
) where {N,K <: MvNormalKernel}
  # CHECK this should be on-manifold for points
  μ_ = mean.(kernels) # This is a ArrayPartition which IS DEFINITELY ON MANIFOLD (we dispatch on mean)
  Σ_ = cov.(kernels)  # on tangent
  
  # FIXME is parallel transport needed here for covariances from different tangent spaces?
  
  _μ, _Σ = if isnothing(μ0)
    calcProductGaussians(M, μ_, Σ_; do_transport_correction)
  else
    calcProductGaussians(M, μ_, Σ_; μ0, do_transport_correction)
  end
  
  return MvNormalKernel(_μ, _Σ, weight)
end




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
