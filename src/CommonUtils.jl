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

# Returns the covariance (square), not deviation
function calcVariableCovarianceBasic(M::AbstractManifold, ptsArr::Vector{P}) where P
  #TODO double check the maths,. it looks like its working at least for groups
  μ = mean(M, ptsArr)
  Xcs = vee.(Ref(M), Ref(μ), log.(Ref(M), Ref(μ), ptsArr))
  Σ = mean(Xcs .* transpose.(Xcs))
  @debug "calcVariableCovarianceBasic" μ
  @debug "calcVariableCovarianceBasic" Σ
  # TODO don't know what to do here so keeping as before, #FIXME it will break
  # a change between this and previous is that a full covariance matrix is returned
  msst = Σ
  msst_ = 0 < sum(1e-10 .< msst) ? maximum(msst) : 1.0
  return msst_
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
  Λ = sum(Λ_)
  
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
