
# see: A Gretton, e.g. http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf

export
  mmd!,  # KED
  mmd

"""
    $SIGNATURES

Normal kernel used for Hilbert space embeddings.
"""
ker(M::MB.AbstractManifold, p, q, sigma::Real=0.001) = exp( -sigma*(distance(M, p, q)^2) )

"""
    $SIGNATURES

MMD disparity (i.e. 'distance') measure based on Kernel Hilbert Embeddings between two beliefs.

Notes:
- This is the in-place version (well more in-place than mmd)

DevNotes:
- TODO dont assume equally weighted particles
- TODO profile SIMD vs SLEEF
- TODO optimize memory
- TODO make multithreaded

See also: [`mmd`](@ref), [`ker`](@ref)
"""
function mmd!(MF::MB.AbstractManifold,
              val::AbstractVector{<:Real},
              a::AbstractVector,
              b::AbstractVector,
              N::Integer=length(a), M::Integer=length(b); 
              bw::AbstractVector{<:Real}=[0.001;] )
  #
  # TODO allow unequal data too
  _N = 1.0/N
  _M = 1.0/M
  _val1 = 0.0
  _val2 = 0.0
  _val3 = 0.0
  @inbounds @fastmath for i in 1:N
    @simd for j in 1:M
      _val1 -= ker(MF, a[i], b[j], bw[1])
    end
  end
  _val1 *= 2.0*_N*_M
  @inbounds @fastmath for i in 1:N
    @simd for j in 1:N
      _val2 += ker(MF, a[i], a[j], bw[1])
    end
  end
  _val2 *= (_N*_N)
  @inbounds @fastmath for i in 1:M
    @simd for j in 1:M
      _val3 += ker(MF, b[i], b[j], bw[1])
    end
  end
  _val3 *= (_M*_M)

  # accumulate all terms
  val[1] = _val1 + _val2 + _val3
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


function mmd(a::ManifoldKernelDensity{M}, b::ManifoldKernelDensity{M}; bw::Vector{<:Real}=[0.001;]) where M <: MB.AbstractManifold
  # @assert a.manifold == b.manifold "Manifolds not the same $(a.manifold), $(b.manifold)"
  aPts = getPoints(a)
  bPts = getPoints(b)
  mmd(a.manifold, aPts, bPts, bw=bw)
end


function isapprox(a::ManifoldKernelDensity, b::ManifoldKernelDensity; atol::Real=0.1)
  mmd(a,b) < atol
end
