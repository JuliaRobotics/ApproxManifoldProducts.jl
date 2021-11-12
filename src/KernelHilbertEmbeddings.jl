
# see: A Gretton, e.g. http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf

export
  mmd!,  # KED
  mmd

"""
    $SIGNATURES

Normal kernel used for Hilbert space embeddings.
"""
ker(M::MB.AbstractManifold, p, q, sigma::Real=0.001) = @fastmath exp( -sigma*(distance(M, p, q)^2) )

# overwrite non-symmetric with alternate implementations 
# ker(M::MB.AbstractManifold, p, q, sigma::Real=0.001) = exp( -sigma*(distance(M, p, q)^2) )


function gramLoops(MF::AbstractManifold, a::AbstractVector, b::AbstractVector, bw::Real)
  
  function _innerLoop(__val::AbstractVector{<:Real},i::Integer)
    # a_ = a[i]
    # __val = [0.0;]
    @inbounds for j in eachindex(b)
      __val .+= ker(MF, a[i], b[j], bw)
    end
    return __val
  end
  
  _val = Threads.Atomic{Float64}(0.0) # 0.0

  # not sure why the mapreduce didnt work.
  # _val -= mapreduce(bj->ker(MF, a[i], bj, bw), -, b)
  Threads.@threads for i in eachindex(a)
    Threads.atomic_add!(_val, _innerLoop([0.0;],i)[1])
  end

  return _val[]
end


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
              bw::AbstractVector{<:Real}=SA[0.001;] )
  #
  # TODO allow unequal data too
  _N = 1.0/N
  _M = 1.0/M

  _val1 = gramLoops(MF, a, b, bw[1])
  _val1 *= -2.0*_N*_M
  
  _val2 = gramLoops(MF, a, a, bw[1])
  _val2 *= (_N^2)

  _val3 = gramLoops(MF, b, b, bw[1])  
  _val3 *= (_M^2)

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


function isapprox(a::ManifoldKernelDensity, b::ManifoldKernelDensity; mmd_tol::Real=1e-1, atol::Real=mmd_tol)
  mmd(a,b) < atol
end
