# define the api for users

export productbelief


# MAYBE deprecate name
manikde!( M::MB.AbstractManifold,
          vecP::AbstractVector{P},
          bw::Union{<:AbstractVector{<:Real},Nothing}=nothing ) where P = ManifoldKernelDensity(M, vecP, vecP[1], bw=bw) 
#


# TODO move to better src file location
isPartial(mkd::ManifoldKernelDensity{M,B,L}) where {M,B,L} = true
isPartial(mkd::ManifoldKernelDensity{M,B,Nothing}) where {M,B} = false


"""
    $SIGNATURES

Approximate the pointwise the product of functionals on manifolds using KernelDensityEstimate.

Notes:
- Always pass full beliefs, for partials use for e.g. `partialDimsWorkaround=[1;3;6]`

Example
-------
```julia
# WARNING, outdated example TODO
using ApproxManifoldProducts

# two densities on a cylinder
p = manikde!(randn(2,100), (:Euclid, :Circular) )

pts2a = 3.0*randn(1,100).+5.0
pts2b = TransformUtils.wrapRad.(0.5*randn(1,100).+pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular))

# convenient plotting (work in progress...)
# TODO update docs

# direct histogram plot
using Gadfly
plot( x=getPoints(pq)[1,:], y=getPoints(pq)[2,:], Geom.histogram2d )
```
"""
function manifoldProduct( ff::AbstractVector{<:ManifoldKernelDensity},
                          mani::M=ff[1].manifold;
                          makeCopy::Bool=false,
                          Niter::Int=1,
                          # partialDimsWorkaround=1:MB.manifold_dimension(mani),
                          ndims::Int=maximum(Ndim.(ff)),
                          N::Int = maximum(Npts.(ff)),
                          u0 = getPoints(ff[1])[1],
                          oldPoints::AbstractVector{P}= [identity(mani, u0) for i in 1:N],
                          addEntropy::Bool=true,
                          recordLabels::Bool=false,
                          selectedLabels::Vector{Vector{Int}}=Vector{Vector{Int}}()) where {M <: MB.AbstractManifold, P}
  #
  # check quick exit
  if 1 == length(ff)
    # @show Ndim(ff[1]), Npts(ff[1]), getPoints(ff[1],false)[1]
    return (makeCopy ? x->deepcopy(x) : x->x)(ff[1])
  end
  
  glbs = KDE.makeEmptyGbGlb();
  glbs.recordChoosen = recordLabels
  
  # TODO DEPRECATE ::NTuple{Symbol} approach
  manif = convert(Tuple,M) #[partialDimsWorkaround]
  addopT, diffopT, getManiMu, _ = buildHybridManifoldCallbacks(manif)


  bws = ones(ndims)
  # MAKE SURE inplace ends up as matrix of coordinates from incoming ::Vector{P}
  oldpts = _pointsToMatrixCoords(mani, oldPoints)
  # FIXME currently assumes oldPoints are in coordinates...
  # @cast oldpts_[i,j] := oldPoints[j][i]
  # oldpts = collect(oldpts_)
  inplace = kde!(oldpts, bws, addopT, diffopT ); # rand(ndims,N)

  # TODO REMOVE
  _ff = (x->x.belief).(ff)
  partialDimMask = Vector{BitVector}(undef, length(ff))
  for (k,md) in enumerate(ff)
    partialDimMask[k] = ones(Int,ndims) .== 1
    if isPartial(md)
      for i in 1:ndims
        if !(i in md._partial)
          partialDimMask[k][i] = false
        end
      end
    end
  end

  ## TODO check both _ff and inplace use a matrix of coordinates (columns)
  # expects Matrix with columns as samples and rows are coordinate dimensions
  pGM, = prodAppxMSGibbsS(inplace, _ff,
                          nothing, nothing, Niter=Niter,
                          partialDimMask=partialDimMask,
                          addop=addopT,
                          diffop=diffopT,
                          getMu=getManiMu,
                          glbs=glbs,
                          addEntropy=addEntropy  );
  #

  if recordLabels
    # how many levels in ball tree
    lc = glbs.labelsChoosen
    nLevels = maximum(keys(lc[1][1]) |> collect)

    # push final label selections onto selectedLabels
    resize!(selectedLabels, N)
    for i in 1:N
      selectedLabels[i] = Int[]
      for j in 1:length(ff)
        push!(selectedLabels[i], lc[i][j][nLevels] - Npts(ff[j]))
      end
    end
  end

  # # if only partials, then keep other dimension values from oldPoints
  # otherDims = ones(ndims) .== 0
  # for msk in partialDimMask
  #   otherDims .|= msk
  # end
  # error(otherDims)

  # build new output ManifoldKernelDensity
  bws[:] = getKDEManifoldBandwidths(pGM, manif)
  bel = kde!(pGM, bws, addopT, diffopT)

  # @error "IN MANIPRODUCT" N size(oldpts) size(pGM,2)

  # @show M
  ManifoldKernelDensity(mani, bel, nothing, ff[1]._u0)
end



# NOTE, this product does not handle combinations of different partial beliefs properly yet
function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct(PP, PP[1].manifold)
end

function *(P1::MKD{M,B}, P2::MKD{M,B}, P_...) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct([P1;P2;P_...], P1.manifold)
end



"""
    $SIGNATURES

Take product of `dens` (including optional partials beliefs) as proposals to be multiplied together.

Notes
-----
- Return points of full dimension, even if only partial dimensions in proposals.
  - 'Other' dimensions left unchanged from incoming `denspts`
- `d` dimensional product approximation
- `partials` are treated per each unique Tuple subgrouping, i.e. (1,2), (2,), ...
- Incorporate ApproxManifoldProducts to process variables in individual batches.

DevNotes
- TODO Consolidate with [`AMP.manifoldProduct`](@ref), especially concerning partials. 
"""
function productbelief( denspts::AbstractVector{P},
                        manifold::MB.AbstractManifold,
                        dens::Vector{<:ManifoldKernelDensity},
                        # partials::Dict{Any, <:AbstractVector{<:ManifoldKernelDensity}},
                        N::Int;
                        asPartial::Bool=false,
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P
  #
  # TODO only works of P <: Vector
  Ndens = length(dens)
  # Npartials = length(partials)
  Ndims = maximum(Ndim.(dens))
  with_logger(logger) do
    @info "[x$(Ndens),d$(Ndims),N$(N)],"
  end
  
  # # resize for #1013
  # if size(denspts,2) < N
  #   pGM = zeros(size(denspts,1),N)
  #   pGM[:,1:size(denspts,2)] .= denspts
  # else
  #   pGM = deepcopy(denspts)
  # end

  mkd = AMP.manifoldProduct(dens, manifold, Niter=1, oldPoints=denspts)
  pGM = getPoints(mkd, asPartial)

  # # TODO VALIDATE inclFull is the right order
  # (pGM, inclFull) = if 0 < Ndens
  #   getPoints(AMP.manifoldProduct(dens, manifold, Niter=1)), true # false
  # elseif Ndens == 0 && 0 < Npartials
  #   deepcopy(denspts), false # true
  # else
  #   error("Unknown density product Ndens=$(Ndens), Npartials=$(Npartials)")
  # end

  # # take the product between partial dimensions
  # _partialProducts!(pGM, partials, manifold, inclFull=inclFull)

  return pGM
end




#
