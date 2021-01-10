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
#
# """
#     $SIGGNATURES
#
# Assemble oplus and ominus operations from given manifols.
#
# Related
#
# buildHybridManifoldCallbacks
# """
# function getManifoldOperations(manis::T) where {T <: Tuple}
#
# end
#

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
