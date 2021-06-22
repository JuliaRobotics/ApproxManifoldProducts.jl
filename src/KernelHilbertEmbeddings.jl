
# see: A Gretton, e.g. http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf

export
  mmd!,  # KED
  mmd


function ker( ::typeof(Euclidean(1)),
              x::AbstractVector{P1},
              y::AbstractVector{P2},
              dx::Vector{<:Real},
              i::Int,
              j::Int;
              sigma::Real=0.001 ) where {P1<:AbstractVector, P2<:AbstractVector}
  #
  dx[1] = x[i][1]
  dx[1] -= y[j][1]
  dx[1] *= dx[1]
  dx[1] *= -sigma
  exp( dx[1] )
end

function ker( ::typeof(Euclidean(2)),
              x::AbstractVector{P1},
              y::AbstractVector{P2},
              dx::Vector{<:Real},
              i::Int,
              j::Int;
              sigma::Real=0.001 ) where {P1<:AbstractVector, P2<:AbstractVector}
  #
  dx[1] = x[i][1]
  dx[2] = x[i][2]
  dx[1] -= y[j][1]
  dx[2] -= y[j][2]
  dx .^= 2
  dx[1] += dx[2]
  dx[1] *= -sigma
  exp( dx[1] )
end

function ker( ::typeof(SE2_Manifold),
              x::AbstractVector{P1},
              y::AbstractVector{P2},
              dx::Vector{<:Real},
              i::Int,
              j::Int;
              sigma::Real=0.001  ) where {P1<:AbstractVector, P2<:AbstractVector}
  #
  innov = se2vee(SE2(x[i][:])\SE2(y[j][:]))
  exp( -sigma*(  innov'*innov  ) )
end

# This functin is still very slow, needs speedup
# Obviously want to get away from the Euler angles throughout
function ker( ::typeof(SE3_Manifold),
              x::AbstractVector{P1},
              y::AbstractVector{P2},
              dx::Vector{<:Real},
              i::Int,
              j::Int;
              sigma::Real=0.001  )  where {P1<:AbstractVector, P2<:AbstractVector}
  #
  innov = veeEuler(SE3(x[i][1:3],Euler((x[i][4:6])...))\SE3(y[j][1:3],Euler((y[j][4:6])...)))
  exp( -sigma*(  innov'*innov  ) )
end


"""
    $SIGNATURES

MMD disparity (i.e. 'distance') measure based on Kernel Hilbert Embeddings between two beliefs.

Notes:
- This is the in-place version (well more in-place than mmd)

DevNotes:
- TODO make work for different sizes N,M
- TODO dont assume equally weighted particles

Related

mmd, ker
"""
function mmd!(val::AbstractVector{<:Real},
              a::AbstractVector{P1},
              b::AbstractVector{P2},
              mani::MF=Euclidean(1),
              N::Int=size(a,2), M::Int=size(b,2); 
              bw::AbstractVector{<:Real}=[0.001;] ) where {P1<:AbstractVector, P2<:AbstractVector, MF <: MB.AbstractManifold}
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

"""
    $SIGNATURES

MMD disparity (i.e. 'distance') measure based on Kernel Hilbert Embeddings between two beliefs.

Notes:
- This is a wrapper to the in-place `mmd!` function.

Related

mmd!, ker
"""
function mmd( a::AbstractVector{P1},
              b::AbstractVector{P2},
              mani::MF=Euclidean(1),
              N::Int=size(a,2), M::Int=size(b,2); 
              bw::AbstractVector{<:Real}=[0.001;]) where {MF <: MB.AbstractManifold, P1<:AbstractVector, P2<:AbstractVector}
  #
  val = [0.0;]
  mmd!( val, a,b,
        mani,
        N, M; bw=bw )
  #
  return val[1]
end


function mmd(a::ManifoldKernelDensity{M}, b::ManifoldKernelDensity{M}; bw::Vector{<:Real}=[0.001;]) where M <: MB.AbstractManifold
  # @assert a.manifold == b.manifold "Manifolds not the same $(a.manifold), $(b.manifold)"
  aPts = getPoints(a)
  bPts = getPoints(b)
  mmd(aPts, bPts, a.manifold, length(aPts), length(bPts), bw=bw)
end


function isapprox(a::ManifoldKernelDensity, b::ManifoldKernelDensity; atol::Real=0.1)
  mmd(a,b) < atol
end
