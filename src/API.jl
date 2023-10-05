# define the api for users



"""
    $SIGNATURES

Approximate the pointwise the product of functionals on manifolds using KernelDensityEstimate.

Notes:
- Always pass full beliefs, for partials use for e.g. `partialDimsWorkaround=[1;3;6]`
- Can also multiply different partials together

Example
-------
```julia
# setup
M = TranslationGroup(3)
N = 75
p = manikde!(M, [randn(3) for _ in 1:N])
q = manikde!(M, [randn(3) .+ 1 for _ in 1:N])

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q])

# direct histogram plot
using Gadfly
plot( x=getPoints(pq)[1,:], y=getPoints(pq)[2,:], Geom.histogram2d )

# TODO, convenient plotting (work in progress...)
```
"""
function manifoldProduct( ff::AbstractVector{<:ManifoldKernelDensity},
                          mani::M=ff[1].manifold;
                          makeCopy::Bool=false,
                          Niter::Integer=1,
                          # partialDimsWorkaround=1:MB.manifold_dimension(mani),
                          ndims::Integer=maximum([0;Ndim.(ff)]),
                          N::Integer = maximum([0;Npts.(ff)]),
                          u0 = getPoints(ff[1], false)[1],
                          oldPoints::AbstractVector{P}= [identity_element(mani, u0) for i in 1:N],
                          addEntropy::Bool=true,
                          recordLabels::Bool=false,
                          selectedLabels::Vector{Vector{Int}}=Vector{Vector{Int}}(),
                          _randU=Vector{Float64}(),
                          _randN=Vector{Float64}(),
                          logger=ConsoleLogger()  ) where {M <: MB.AbstractManifold, P}
  #
  # check quick exit
  if  1 == length(ff)
    # @show Ndim(ff[1]), Npts(ff[1]), getPoints(ff[1],false)[1]
    return (makeCopy ? x->deepcopy(x) : x->x)(ff[1])
  end

  # TODO DEPRECATE ::NTuple{Symbol} approach
  manif = convert(Tuple, mani)  #[partialDimsWorkaround]
  addopT, diffopT, getManiMu, _ = buildHybridManifoldCallbacks(manif)
  
  Ndens = length(ff)
  # Npartials = length(partials)
  Ndims = maximum([0;Ndim.(ff)])
  with_logger(logger) do
    @debug "[x$(Ndens),d$(Ndims),N$(N)],"
  end
  
  glbs = KDE.makeEmptyGbGlb();
  glbs.recordChoosen = recordLabels
  
  bws = 0 == length(ff) ? [1.0;] : ones(ndims)
  # MAKE SURE inplace ends up as matrix of coordinates from incoming ::Vector{P}
  oldpts = _pointsToMatrixCoords(mani, oldPoints)
  # FIXME currently assumes oldPoints are in coordinates...
  # @cast oldpts_[i,j] := oldPoints[j][i]
  # oldpts = collect(oldpts_)
  inplace = kde!(oldpts, bws, addopT, diffopT ); # rand(ndims,N)

  # TODO refactor and reduce
  _u0 = 0 == length(ff) ? u0 : ff[1]._u0
  if 0 == length(ff)
    return ManifoldKernelDensity(mani, inplace, zeros(manifold_dimension(mani)) .== 0, _u0)
  end

  _ff = map(x->x.belief, ff)
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
  
  ndims = maximum([0;Ndim.(_ff)])
  Ndens = length(_ff)
  Np    = Npts(inplace)
  maxNp = maximum(Int[Np; Npts.(_ff)])
  Nlevels = floor(Int,(log(Float64(maxNp))/log(2.0))+1.0)
  if 0 == length(_randU)
    _len = Int(Np*Ndens*(Niter+2)*Nlevels)
    resize!(_randU, _len)
    _randU .= rand(_len)
  end
  if 0 == length(_randN)
    _len = Int(ndims*Np*(Nlevels+1))
    resize!(_randN, _len)
    _randN .= randn(_len) 
  end

  ## TODO check both _ff and inplace use a matrix of coordinates (columns)
  # expects Matrix with columns as samples and rows are coordinate dimensions
  pGM, = prodAppxMSGibbsS(inplace, _ff,
                          nothing, nothing; 
                          Niter,
                          partialDimMask,
                          addop=addopT,
                          diffop=diffopT,
                          getMu=getManiMu,
                          glbs,
                          addEntropy,
                          ndims,
                          Ndens,
                          Np,
                          maxNp,
                          Nlevels,
                          randU=_randU,
                          randN=_randN  );
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
        push!(selectedLabels[i], lc[i][j][nLevels])
      end
    end
  end

  # if only partials, then keep other dimension values from oldPoints
  otherDims = ones(ndims) .== 0
  for msk in partialDimMask
    otherDims .|= msk
  end

  # build new output ManifoldKernelDensity
  bws[:] = getKDEManifoldBandwidths(pGM, manif)
  bel = kde!(pGM, bws, addopT, diffopT)
  
  # FIXME u0 might not be representative of the partial information
  return ManifoldKernelDensity(mani, bel, otherDims, _u0)
end



# NOTE, this product does not handle combinations of different partial beliefs properly yet
function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct(PP, PP[1].manifold)
end

function *(P1::MKD{M,B}, P2::MKD{M,B}, P_...) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct([P1;P2;P_...], P1.manifold)
end






#
