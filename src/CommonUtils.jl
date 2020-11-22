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
