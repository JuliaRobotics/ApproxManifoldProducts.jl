# test KernelEmbeddingDistribution

using BenchmarkTools
using Distributions
using ApproxManifoldProducts
import ApproxManifoldProducts: mmd!
# using StaticArrays
# using KernelDensityEstimate


function main()
  P = randn(100)
  Q = randn(100)
  # v = StaticArray(0.0,)
  v = Float64[0.0]

  @btime mmd!(v, P, Q)

  nothing
end

main()


function main2()

  P1_ = kde!(randn(100))
  Q1 = kde!(randn(100))

  @btime kld( P1_, Q1)
  nothing
end

main2()


#
