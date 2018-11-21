# test KernelEmbeddingDistribution

using BenchmarkTools
using Distributions
# using StaticArrays

ker(x::Real,y::Real; sigma::Float64=2.0)::Float64 = exp( -sigma*((x-y)^2) )

# Assuming equally weighted particles
function mmd!(val::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64})
  len = length(a)
  @assert len == length(b)
  reci_len = 1.0/len
  val = 0.0
  @inbounds @fastmath for i in 1:len
     @simd for j in 1:len
       val -= ker(a[i], b[j])
     end
  end
  @fastmath val *= 2.0
  @inbounds @fastmath for i in 1:len
    @simd for j in 1:len
      val += ker(a[i], a[j])
      val += ker(b[i], b[j])
    end
  end
  @fastmath val *= reci_len
  @fastmath val *= reci_len
  return val
end


using KernelDensityEstimate

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
