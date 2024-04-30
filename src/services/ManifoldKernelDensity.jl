
## ==========================================================================================
## helper functions to contruct MKD objects
## ==========================================================================================


ManifoldKernelDensity(
  mani::M, 
  bel::B, 
  ::Nothing=nothing, 
  u0::P=zeros(manifold_dimension(mani));
  infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) 
) where {M <: MB.AbstractManifold, B <: BallTreeDensity, P} = ManifoldKernelDensity{M,B,Nothing,P}(
  mani,
  bel,
  nothing,
  u0,
  infoPerCoord
)
#

function ManifoldKernelDensity( 
  mani::M, 
  bel::B, 
  partial::L, 
  u0::P=zeros(manifold_dimension(mani));
  infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0)) 
) where {M <: MB.AbstractManifold, B <: BallTreeDensity, L<:AbstractVector{<:Integer}, P}
  #
  if length(partial) != manifold_dimension(mani)
    # call the constructor direct
    return ManifoldKernelDensity{M,B,L,P}(mani,bel,partial,u0,infoPerCoord)
  else
    # full manifold, therefore equivalent to L::Nothing
    return ManifoldKernelDensity(mani,bel,nothing,u0;infoPerCoord=infoPerCoord)
  end
end

ManifoldKernelDensity(
  mani::M, 
  bel::B, 
  pl_mask::Union{<:BitVector,<:AbstractVector{<:Bool}}, 
  u0::P=zeros(manifold_dimension(mani));
  infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(mani, u0))
) where {M <: MB.AbstractManifold, B <: BallTreeDensity, P} = ManifoldKernelDensity(
  mani,
  bel,
  (1:manifold_dimension(mani))[pl_mask],
  u0;
  infoPerCoord
)


function ManifoldKernelDensity( 
  M::MB.AbstractManifold,
  vecP::AbstractVector{P},
  u0 = vecP[1],
  ϵ = identity_element(M, u0); # vecP[1]
  partial::L=nothing,
  infoPerCoord::AbstractVector{<:Real}=ones(getNumberCoords(M, u0)),
  dims::Int=manifold_dimension(M),
  bw::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real},Nothing}=nothing,
  belmodel::Function = (a,b,aF,dF) -> KernelDensityEstimate.kde!(a, collect(b), aF, dF) # collect(b) but error length(::Nothing)
) where {P,L}
  #
  # FIXME obsolete
  arr = Matrix{Float64}(undef, dims, length(vecP))
  
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
  bel = belmodel(arr,_bw,addopT,diffopT)
  # bel = KernelDensityEstimate.kde!(arr, collect(_bw), addopT, diffopT)
  return ManifoldKernelDensity(M, bel, partial, u0, infoPerCoord)
end


# MAYBE deprecate name
manikde!( 
  M::MB.AbstractManifold,
  vecP::AbstractVector{P},
  u0::P=vecP[1];
  kw... 
) where P = ManifoldKernelDensity(
  M, 
  vecP, 
  u0; 
  kw...
)

#

function manikde!_manellic(
  M::AbstractManifold,
  pts::AbstractVector;
  bw=ones(manifold_dimension(M),1),
)
  #

  mtree = ApproxManifoldProducts.buildTree_Manellic!(
    M,
    pts;
    kernel_bw=bw, 
    kernel=AMP.MvNormalKernel
  )
  
  # Cost function to optimize
  _cost(_pts, σ) = begin
    # FIXME avoid rebuilding tree at each optim iteration!!!
    mtr = buildTree_Manellic!(M, _pts; kernel_bw=reshape([σ;],manifold_dimension(M),1), kernel=MvNormalKernel)
    entropy(mtr)
  end
  
  # optimize for best LOOCV bandwidth
  # FIXME switch to RLM (or other Manopt) techinque instead 
  # set lower and upper bounds for Golden section optimization
  lcov, ucov = getBandwidthSearchBounds(mtree)
  res = Optim.optimize(
    (s)->_cost(pts,s^2), 
    lcov[1], ucov[1], Optim.GoldenSection()
  )
  best_cov = [Optim.minimizer(res);;]

  # return tree with correct bandwidth
  # TODO avoid tree rebuild somehow
  manikde!(
    M,
    pts;
    bw=best_cov,
    belmodel = (a,b,aF,dF) -> ApproxManifoldProducts.buildTree_Manellic!(
      M,
      pts;
      kernel_bw=b, 
      kernel=AMP.MvNormalKernel
    )
  )
end


## ==========================================================================================
## a few utilities
## ==========================================================================================


_getFieldPartials(mkd::ManifoldKernelDensity{M,B,Nothing}, field::Function, aspartial::Bool=true) where {M,B} = field(mkd)

function _getFieldPartials(mkd::ManifoldKernelDensity{M,B,<:AbstractVector}, field::Function, aspartial::Bool=true) where {M,B}
  val = field(mkd)
  if aspartial && (length(val) == length(mkd._partial))
    return val
  elseif !aspartial && (length(val) == length(mkd._partial))
    val_ = zeros(manifold_dimension(mkd.manifold))
    val_[mkd._partial] .= val
    return val_
  elseif aspartial && (length(val) == manifold_dimension(mkd.manifold))
    return val[mkd._partial]
  elseif !aspartial && (length(val) == manifold_dimension(mkd.manifold))
    return val
  else
    error("unknown size MKD.$(field) with partial length=$(length(mkd._partial)) vs length=$(length(val)) --- and value=$val")
  end
end



getInfoPerCoord(mkd::ManifoldKernelDensity, aspartial::Bool=true) = _getFieldPartials(mkd, x->x.infoPerCoord, aspartial)

getBandwidth(mkd::ManifoldKernelDensity, aspartial::Bool=true) = _getFieldPartials(mkd, x->getBW(x)[:,1], aspartial)


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


function getBW(x::ManifoldKernelDensity{M,B,L}, asPartial::Bool=true; kw...) where {M,B,L}
  bw = getBW(x.belief; kw...)
  if L !== Nothing && asPartial
    return view(bw, x._partial)
  end
  return bw
end


# TODO check that partials / marginals are sampled correctly
function sample(x::ManifoldKernelDensity{M,B,L,P}, N::Integer=1) where {M,B,L,P}
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
  pts = if N < Npts(x)
    # get points with non-partial coord dims so that new MKD can be built
    shuffle(getPoints(x, false))[1:N]
  else
    _pts, = sample(x, N)
    _pts
  end
  ManifoldKernelDensity(x.manifold, pts, x._u0, partial=x._partial, infoPerCoord=x.infoPerCoord)
end


function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B,L,P}) where {M,B,L,P}
  _round(s::AbstractArray; kw...) = round.(s[:];kw...)
  _round(s::AbstractVector{<:AbstractMatrix}; kw...) = round.(s[1][:];kw...)

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
  println(io, "  bws:   ", getBandwidth(mkd, true) |> x->_round(x;digits=4)) # .|> x->round(x,digits=4))
  println(io, "  ipc:   ", getInfoPerCoord(mkd, true) .|> x->round(x,digits=4))
  print(io, "   mean: ")
  try
    # mn = mean(mkd.manifold, getPoints(mkd, false))
    mn = mean(mkd)
    if mn isa ProductRepr
      println(io)
      for prt in mn.parts
        println(io, "         ", round.(prt,digits=4))
      end
    else
      println(io, round.(mn',digits=4))
    end
  catch
    println(io, "----")
  end
  println(io, ")")
  nothing
end


Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)
Base.show(io::IO, ::MIME"application/juno.inline", mkd::ManifoldKernelDensity) = show(io, mkd)


# override
function marginal(
  x::ManifoldKernelDensity{M,B}, 
  dims::AbstractVector{<:Integer}
) where {M <: AbstractManifold, B}
  #
  ldims::Vector{Int} = collect(dims)
  ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end

function marginal(
  x::ManifoldKernelDensity{M,B,L}, 
  dims::AbstractVector{<:Integer}
) where {M <: AbstractManifold , B, L <: AbstractVector{<:Integer}}
  #
  ldims::Vector{Int} = intersect(x._partial, dims)
  ManifoldKernelDensity(x.manifold, x.belief, ldims, x._u0)
end
# manis = convert(Tuple, x.manifold)
# partMani = _reducePartialManifoldElements(manis[dims])
# pts = getPoints(x)


"""
    $SIGNATURES

If a marginal (statistics) of a probability reduce the dimensions (i.e. casts a shadow, or projects); then an 
antimarginal aims to increase dimension of the probability within reason.

Notes
- Marginalization is a integration of for higher dimension to lower dimension, so antimarginal likely involve 
  differentiation (anti-integral) instead.
- In manifold language, this is an embedding into a higher dimensional space.
- See structure from motion in machine vision, or stereo disparity for building depth clouds from 2D images.
- Imagine combining three different partial embedding A=[1, nan, nan, 1.4], B=[nan,2.1,nan,4], C=[nan, nan, 3, nan]
  which should equal A+B+C = [notnan, notnan, notnan, notnan]
"""
function antimarginal(newM::AbstractManifold,
                      u0,
                      mkd::ManifoldKernelDensity, 
                      newpartial::AbstractVector{<:Integer} )
  #

  # convert to antimarginal by copying user provided example point for bigger manifold
  pts = getPoints(mkd, false)
  # new coord partials must be placed into a full dimension point, thats why we use u0
  nPts = Vector{typeof(u0)}(undef, length(pts))
  for i in eachindex(pts)
    setPointPartial!(newM, nPts, mkd.manifold, pts, newpartial, i)
  end

  # also update metadata elements
  finalpartial = !isPartial(mkd) ? newpartial : error("not built yet, to shift incoming partial")
  bw = zeros(manifold_dimension(newM))
  bw[finalpartial] .= getBW(mkd)[:,1]
  ipc = zeros(manifold_dimension(newM))
  ipc[finalpartial] .= getInfoPerCoord(mkd, true)
  
  manikde!(newM, nPts, u0; bw, partial=finalpartial, infoPerCoord=ipc)
end


## =======================================================================================
##  deprecate as necessary below
## =======================================================================================

Base.convert(::Type{B}, mkd::ManifoldKernelDensity{M,B}) where {M,B<:BallTreeDensity} = mkd.belief



#
