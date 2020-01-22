
# abstract type ManifoldDefs end
#
struct Euclid <: Manifold end
# struct Euclid2 <: Manifold end
# struct SE2_Manifold <: Manifold end

function ker(::Type{Euclid},
             x::Array{<:Real,2},
             y::Array{<:Real,2},
             dx::Vector{<:Real},
             i::Int,
             j::Int;
             sigma::Float64=2.0)::Float64
  #
  dx[1] = x[1,i]
  dx[1] -= y[1,j]
  dx[1] *= dx[1]
  dx[1] *= -sigma
  exp( dx[1] )
end

function ker(::Type{Euclid2},
             x::Array{<:Real,2},
             y::Array{<:Real,2},
             dx::Vector{<:Real},
             i::Int,
             j::Int;
             sigma::Float64=2.0  )::Float64
  #
  dx[1] = x[1,i]
  dx[2] = x[2,i]
  dx[1] -= y[1,j]
  dx[2] -= y[2,j]
  dx .^= 2
  dx[1] += dx[2]
  dx[1] *= -sigma
  exp( dx[1] )
end

# function ker(::Type{SE2_Manifold}, x::Vector{<:Real},y::Vector{<:Real}; sigma::Float64=2.0)::Float64
#   innov = se2vee(SE2(y)\SE2(x))
#   exp( -sigma*(  innov'*innov  ) )
# end


# Assuming equally weighted particles
function mmd!(val::Vector{Float64}, a::Array{Float64,2}, b::Array{Float64,2}, mani::Type{<:Manifold}=Euclid, N::Int=size(a,2), M::Int=size(b,2); bw::Vector{Float64}=[2.0;] )
  # TODO allow unequal data too
  @assert N == M
  # reci_len = 1.0/N
  val[1] = 0.0
  dx = zeros(2)
  @inbounds @fastmath for i in 1:N
     @simd for j in 1:M
       val[1] -= ker(mani, a, b, dx, i, j, sigma=bw[1])
     end
  end
  val .*= 2.0
  @inbounds @fastmath for i in 1:N
    @simd for j in 1:M
      val[1] += ker(mani, a, a, dx, i, j, sigma=bw[1])
      val[1] += ker(mani, b, b, dx, i, j, sigma=bw[1])
    end
  end
  val ./= N
  val ./= M
  return val
end

# mmd!(val::Vector{Float64}, a::Array{Float64,1}, b::Array{Float64,1}, mani::Type{<:Manifold}=Euclid) = mmd!( val, reshape(a,1,:), reshape(b,1,:), mani )



##
#
# N = 10000
# a = randn(2,N)
# b = randn(2,N)
# c = randn(2,N)
# c[1,:] .+= 10
#
# res = zeros(1)
#
# @time mmd!(res, a,b, Euclid2)
