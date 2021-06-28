# define the api for users


manikde!( ptsArr::AbstractVector{P}, 
          M::MB.AbstractManifold  ) where P <: AbstractVector = ManifoldKernelDensity(M, ptsArr) 
#

manikde!( ptsArr::AbstractVector{P}, 
          bw::AbstractVector{<:Real}, 
          M::MB.AbstractManifold  ) where P <: AbstractVector = ManifoldKernelDensity(M, ptsArr, bw)
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
                          oldPoints::AbstractVector{P}=[randn(ndims) for _ in 1:N],
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
  @cast oldpts_[i,j] := oldPoints[j][i]
  oldpts = collect(oldpts_)
  dummy = kde!(oldpts, bws, addopT, diffopT ); # rand(ndims,N)

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
  pGM, = prodAppxMSGibbsS(dummy, _ff,
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
  # @show M
  ManifoldKernelDensity(mani,bel)
end




#
