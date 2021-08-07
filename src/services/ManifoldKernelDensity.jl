
import Random: rand

export getPoints, getBW, Ndim, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export calcMean
export antimarginal


## ==========================================================================================
## helper functions to contruct MKD objects
## ==========================================================================================


ManifoldKernelDensity(mani::M, 
                      bel::B, 
                      ::Nothing=nothing, 
                      u0::P=zeros(manifold_dimension(mani));
                      infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) ) where {M <: MB.AbstractManifold, B <: BallTreeDensity, P} = ManifoldKernelDensity{M,B,Nothing,P}(mani,bel,nothing,u0,infoPerCoord)
#

function ManifoldKernelDensity( mani::M, 
                                bel::B, 
                                partial::L, 
                                u0::P=zeros(manifold_dimension(mani));
                                infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) ) where {M <: MB.AbstractManifold, B <: BallTreeDensity, L<:AbstractVector{<:Integer}, P}
  #
  if length(partial) != manifold_dimension(mani)
    # call the constructor direct
    return ManifoldKernelDensity{M,B,L,P}(mani,bel,partial,u0,infoPerCoord)
  else
    # full manifold, therefore equivalent to L::Nothing
    return ManifoldKernelDensity(mani,bel,nothing,u0;infoPerCoord=infoPerCoord)
  end
end

ManifoldKernelDensity(mani::M, 
                      bel::B, 
                      pl_mask::Union{<:BitVector,<:AbstractVector{<:Bool}}, 
                      u0::P=zeros(manifold_dimension(mani));
                      infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) ) where {M <: MB.AbstractManifold, B <: BallTreeDensity, P} = ManifoldKernelDensity(mani,bel,(1:manifold_dimension(mani))[pl_mask],u0;infoPerCoord=infoPerCoord)


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
  ϵ = identity_element(M, vecP[1])

  for j in 1:length(vecP)
    arr[:,j] = vee(M, ϵ, log(M, ϵ, vecP[j]))
  end

  manis = convert(Tuple, M)
  # find or have the bandwidth
  _bw = bw === nothing ? getKDEManifoldBandwidths(arr, manis ) : bw
  # NOTE workaround for partials and user did not specify a bw
  if bw === nothing && partial !== nothing
    mask = ones(Int, length(_bw)) .== 1
    mask[partial] .= false
    _bw[mask] .= 1.0
  end
  addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
  bel = KernelDensityEstimate.kde!(arr, _bw, addopT, diffopT)
  return ManifoldKernelDensity(M, bel, partial, u0, infoPerCoord)
end


# MAYBE deprecate name
manikde!( M::MB.AbstractManifold,
          vecP::AbstractVector{P},
          u0::P=vecP[1];
          kw... ) where P = ManifoldKernelDensity(M, vecP, u0; kw...) 
#


## ==========================================================================================
## a few utilities
## ==========================================================================================

function Statistics.mean(mkd::ManifoldKernelDensity, aspartial::Bool=true)
  M = if aspartial && isPartial(mkd)
    getManifoldPartial(mkd.manifold, mkd._partial)
  else
    mkd.manifold
  end

  mean(mkd.manifold, getPoints(mkd, aspartial))
end

"""
    $SIGNATURES

Alias for overloaded `Statistics.mean`.
"""
calcMean(mkd::ManifoldKernelDensity, aspartial::Bool=true) = mean(mkd, aspartial)



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

"""
    $SIGNATURES

Return true if this ManifoldKernelDensity is a partial.
"""
isPartial(mkd::ManifoldKernelDensity{M,B,L}) where {M,B,L} = true
isPartial(mkd::ManifoldKernelDensity{M,B,Nothing}) where {M,B} = false
  
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
  # @show dims x._partial
  ldims::Vector{Int} = intersect(x._partial, dims)
  ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end
# manis = convert(Tuple, x.manifold)
# partMani = _reducePartialManifoldElements(manis[dims])
# pts = getPoints(x)

getInfoPerCoord(mkd::ManifoldKernelDensity{M,B,Nothing}, ::Bool=true) where {M,B} = mkd.infoPerCoord
function getInfoPerCoord(mkd::ManifoldKernelDensity{M,B,<:AbstractVector}, aspartial::Bool=true) where {M,B}
  ipc = mkd.infoPerCoord
  if aspartial && (length(ipc) == length(mkd._partial))
    return ipc
  elseif !aspartial && (length(ipc) == length(mkd._partial))
    ipc_ = zeros(manifold_dimension(mkd.manifold))
    ipc_[mkd._partial] .= ipc
    return ipc_
  elseif aspartial && (length(ipc) == manifold_dimension(mkd.manifold))
    return ipc[mkd._partial]
  elseif !aspartial && (length(ipc) == manifold_dimension(mkd.manifold))
    return ipc
  else
    error("unknown infoPerCoord length=$(length(ipc)) vs. partial length=$(length(mkd._partial))")
  end
end

function antimarginal(newM::AbstractManifold,
                      u0,
                      mkd::ManifoldKernelDensity, 
                      newpartial::AbstractVector{<:Integer} )
  #

  # convert to antimarginal by copying user provided example point for bigger manifold
  pts = getPoints(mkd, false)
  nPts = Vector{typeof(u0)}(undef, length(pts))
  for (i,pt) in enumerate(pts)
    nPts[i] = setPointsManiPartial!(newM, deepcopy(u0), mkd.manifold, pt, newpartial)
  end

  # also update metadata elements
  finalpartial = !isPartial(mkd) ? newpartial : error("not built yet, to shift incoming partial")
  bw = zeros(manifold_dimension(newM))
  bw[finalpartial] .= getBW(mkd)[:,1]
  ipc = zeros(manifold_dimension(newM))
  ipc[finalpartial] .= getInfoPerCoord(mkd, true)
  
  manikde!(newM, nPts, u0, bw=bw, partial=finalpartial, infoPerCoord=ipc)
end

"""
    $SIGNATURES

Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

Notes
- Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
- Second argument controls whether partial dimensions only should be returned (`=true` default).

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

Random.rand(mkd::ManifoldKernelDensity, N::Integer=1) = sample(mkd, N)[1]



function resample(x::ManifoldKernelDensity, N::Int)
  pts, = sample(x, N)
  ManifoldKernelDensity(x.manifold, pts, x._u0, partial=x._partial)
end


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
  print(io, "  dims:  ", Ndim(mkd.belief))
  printstyled(io, isPartial(mkd) ? "* --> $(length(mkd._partial))" : "", bold=true)
  println(io)
  println(io, "  prtl:   ", mkd._partial)
  bw = getBW(mkd.belief)[:,1]
  pvec = isPartial(mkd) ? mkd._partial : collect(1:length(bw))
  println(io, "  bws:   ", bw[pvec] .|> x->round(x,digits=4))
  println(io, "  ipc:   ", mkd.infoPerCoord[pvec] .|> x->round(x,digits=4))
  try
    # mn = mean(mkd.manifold, getPoints(mkd, false))
    mn = mean(mkd)
    println(io, "   mean: ", round.(mn',digits=4))
  catch
  end
  println(io, ")")
  nothing
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)
Base.show(io::IO, ::MIME"application/juno.inline", mkd::ManifoldKernelDensity) = show(io, mkd)

## =======================================================================================
##  deprecate as necessary below
## =======================================================================================

Base.convert(::Type{B}, mkd::ManifoldKernelDensity{M,B}) where {M,B<:BallTreeDensity} = mkd.belief



#
