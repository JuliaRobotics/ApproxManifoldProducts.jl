
# see: A Gretton, e.g. http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf

export
  mmd!,  # KED
  mmd


ker(M::MB.AbstractManifold, p, q, sigma::Real=0.001) = exp( -sigma*(distance(M, p, q)^2) )

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
function mmd!(MF::MB.AbstractManifold,
              val::AbstractVector{<:Real},
              a::AbstractVector,
              b::AbstractVector,
              N::Int=length(a), M::Int=length(b); 
              bw::AbstractVector{<:Real}=[0.001;] )
  #
  # TODO allow unequal data too
  @assert N == M "mmd! currently requires input vectors be the same length"
  val[1] = 0.0
  @inbounds @fastmath for i in 1:N
    @simd for j in 1:M
      val[1] -= ker(MF, a[i], b[j], bw[1])
    end
  end
  val .*= 2.0
  @inbounds @fastmath for i in 1:N
    @simd for j in 1:M
      val[1] += ker(MF, a[i], a[j], bw[1])
      val[1] += ker(MF, b[i], b[j], bw[1])
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
function mmd( MF::MB.AbstractManifold,
              a::AbstractVector,
              b::AbstractVector,
              N::Int=length(a), M::Int=length(b); 
              bw::AbstractVector{<:Real}=[0.001;])
  #
  val = [0.0;]
  mmd!( MF, val, 
        a,b,
        N, M; bw=bw )
  #
  return val[1]
end

@deprecate mmd!(v::AbstractVector{<:Real}, a::AbstractArray,b::AbstractArray,MF::MB.AbstractManifold, w...; kw...) mmd!(MF, v, a, b, w...; kw...)

@deprecate mmd(a::AbstractArray,b::AbstractArray,MF::MB.AbstractManifold, w...; kw...) mmd(MF, a, b, w...; kw...)

function mmd(a::ManifoldKernelDensity{M}, b::ManifoldKernelDensity{M}; bw::Vector{<:Real}=[0.001;]) where M <: MB.AbstractManifold
  # @assert a.manifold == b.manifold "Manifolds not the same $(a.manifold), $(b.manifold)"
  aPts = getPoints(a)
  bPts = getPoints(b)
  mmd(a.manifold, aPts, bPts, bw=bw)
end


function isapprox(a::ManifoldKernelDensity, b::ManifoldKernelDensity; atol::Real=0.1)
  mmd(a,b) < atol
end
