
import Random: rand

export getPoints, getBW, Ndim, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export calcMean

function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B,L,P}) where {M,B,L,P}
  printstyled(io, "ManifoldKernelDensity{", bold=true, color=:blue )
  println(io)
  printstyled(io, "    M", bold=true, color=:magenta )
  print(io, " = ", M, ",")
  println(io)
  printstyled(io, "    B", bold=true, color=:magenta )
  print(io, " = ", B, ",")
  println(io)
  printstyled(io, "    L", bold=true, color=:magenta )
  print(io, " = ", L, ",")
  println(io)
  printstyled(io, "    P", bold=true, color=:magenta )
  print(io, " = ", P)
  println(io)
  println(io, " }(")
  println(io, "  Npts:  ", Npts(mkd.belief))
  println(io, "  dims:  ", Ndim(mkd.belief))
  println(io, "  prtl:  ", mkd._partial)
  println(io, "  bws:   ", getBW(mkd.belief)[:,1] .|> x->round(x,digits=4))
  println(io, "  ipc:   ", mkd.infoPerCoord .|> x->round(x,digits=4))
  try
    mn = mean(mkd.manifold, getPoints(mkd, false))
    println(io, "   mean:  ", round.(mn',digits=4))
  catch
  end
  println(io, ")")
  nothing
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)



ManifoldKernelDensity(mani::M, 
                      bel::B, 
                      partial::L=nothing, 
                      u0::P=zeros(manifold_dimension(mani));
                      infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) ) where {M <: MB.AbstractManifold, B <: BallTreeDensity, L, P} = ManifoldKernelDensity{M,B,L,P}(mani,bel,partial,u0,infoPerCoord)
#


function ManifoldKernelDensity( M::MB.AbstractManifold,
                                vecP::AbstractVector{P},
                                u0=vecP[1];
                                partial::L=nothing,
                                infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(M, u0)),
                                dims::Int=manifold_dimension(M),
                                bw::Union{<:AbstractVector{<:Real},Nothing}=nothing  ) where {P,L}
  #
  # FIXME obsolete
  arr = Matrix{Float64}(undef, dims, length(vecP))
  ϵ = identity(M, vecP[1])

  for j in 1:length(vecP)
    arr[:,j] = vee(M, ϵ, log(M, ϵ, vecP[j]))
  end

  manis = convert(Tuple, M)
  # find or have the bandwidth
  _bw = bw === nothing ? getKDEManifoldBandwidths(arr, manis ) : bw
  addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
  bel = KernelDensityEstimate.kde!(arr, _bw, addopT, diffopT)
  return ManifoldKernelDensity(M, bel, partial, u0, infoPerCoord)
end


# MAYBE deprecate name
manikde!( M::MB.AbstractManifold,
          vecP::AbstractVector{P},
          w...;
          kw... ) where P = ManifoldKernelDensity(M, vecP, w...; kw...) 
#


# internal workaround function for building partial submanifold dimensions, must be upgraded/standarized
function _buildManifoldPartial( fullM::MB.AbstractManifold, 
                                partial_coord_dims )
  #
  # temporary workaround during Manifolds.jl integration
  manif = convert(Tuple, fullM)[partial_coord_dims]
  # 
  newMani = MB.AbstractManifold[]
  for me in manif
    push!(newMani, _reducePartialManifoldElements(me))
  end

  # assume independent dimensions for definition, ONLY USED AS DECORATOR AT THIS TIME, FIXME
  return ProductManifold(newMani...)
end

# override
function marginal(x::ManifoldKernelDensity{M,B}, 
  dims::AbstractVector{<:Integer}  ) where {M <: AbstractManifold , B}
#
ldims::Vector{Int} = collect(dims)
ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end

function marginal(x::ManifoldKernelDensity{M,B,L}, 
  dims::AbstractVector{<:Integer}  ) where {M <: AbstractManifold , B, L <: AbstractVector{<:Integer}}
#
ldims::Vector{Int} = collect(L[dims])
ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end
# manis = convert(Tuple, x.manifold)
# partMani = _reducePartialManifoldElements(manis[dims])
# pts = getPoints(x)


"""
    $SIGNATURES

Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

Notes
- Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
- Second argument controls whether partial dimensions only should be returned.

DevNotes
- Currently converts down to manifold from matrix of coordinates (legacy), to be deprecated TODO
"""
function getPoints(x::ManifoldKernelDensity{M,B}, ::Bool=true) where {M <: AbstractManifold, B}
  _matrixCoordsToPoints(x.manifold, getPoints(x.belief), x._u0)
end

function getPoints( x::ManifoldKernelDensity{M,B,L}, 
                    aspartial::Bool=true) where {M <: AbstractManifold, B, L <: AbstractVector{Int}}
  #
  pts = getPoints(x.belief)
  
  (M_,pts_, u0_) = if (L !== nothing) && aspartial
    Mp, Rp = getManifoldPartial(x.manifold, x._partial, x._u0)
    (Mp, view(pts, x._partial, :), Rp)
  else
    (x.manifold, pts, x._u0)
  end

  _matrixCoordsToPoints(M_, pts_, u0_)
end


# TODO check that partials / marginals are sampled correctly
function sample(x::ManifoldKernelDensity{M,B,L,P}, N::Int) where {M,B,L,P}
  # get legacy matrix of coordinates and selected labels
  coords, lbls = sample(x.belief, N)

  # pack samples into vector of point type P
  vecP = Vector{P}(undef, N)
  for j in 1:N
    vecP[j] = makePointFromCoords(x.manifold, view(coords, :, j), x._u0)
  end

  vecP, lbls
end


function resample(x::ManifoldKernelDensity, N::Int)
  pts, = sample(x, N)
  ManifoldKernelDensity(x.manifold, pts, x._u0, partial=x._partial)
end


## =======================================================================================
##  deprecate as necessary below
## =======================================================================================

Base.convert(::Type{B}, mkd::ManifoldKernelDensity{M,B}) where {M,B<:BallTreeDensity} = mkd.belief



function calcMean(mkd::ManifoldKernelDensity{M}) where {M <: ManifoldsBase.AbstractManifold}
  data = getPoints(mkd)
  # Returns the mean point on manifold for consitency
  mean(mkd.manifold, data)  
end


#