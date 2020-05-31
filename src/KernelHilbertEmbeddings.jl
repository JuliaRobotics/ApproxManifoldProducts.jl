
# see: A Gretton, e.g. http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf

# abstract type ManifoldDefs end
#
struct Euclid <: Manifold end
struct SE2_Manifold <: Manifold end

function ker(::Type{Euclid},
             x::AbstractArray{<:Real,2},
             y::AbstractArray{<:Real,2},
             dx::Vector{<:Real},
             i::Int,
             j::Int;
             sigma::Float64=2.0 )
  #
  dx[1] = x[1,i]
  dx[1] -= y[1,j]
  dx[1] *= dx[1]
  dx[1] *= -sigma
  SLEEFPirates.exp( dx[1] )
end

function ker(::Type{Euclid2},
             x::AbstractArray{<:Real,2},
             y::AbstractArray{<:Real,2},
             dx::Vector{<:Real},
             i::Int,
             j::Int;
             sigma::Float64=2.0 )
  #
  dx[1] = x[1,i]
  dx[2] = x[2,i]
  dx[1] -= y[1,j]
  dx[2] -= y[2,j]
  dx .^= 2
  dx[1] += dx[2]
  dx[1] *= -sigma
  SLEEFPirates.exp( dx[1] )
end

function ker(::Type{SE2_Manifold},
             x::AbstractMatrix{<:Real},
             y::AbstractMatrix{<:Real},
             dx::Vector{<:Real},
             i::Int,
             j::Int;
             sigma::Float64=0.001  )
  #
  innov = se2vee(SE2(x[:,i])\SE2(y[:,j]))
  SLEEFPirates.exp( -sigma*(  innov'*innov  ) )
end


# Assuming equally weighted particles
# TODO make work for different sizes
function mmd!(val::AbstractVector{<:Real},
              a::AbstractArray{<:Real,2},
              b::AbstractArray{<:Real,2},
              mani::Type{<:Manifold}=Euclid,
              N::Int=size(a,2), M::Int=size(b,2); bw::Vector{Float64}=[2.0;] )
  #
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
