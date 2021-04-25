# define the api for users

"""
    $SIGNATURES

Approximate the pointwise the product of functionals on manifolds using KernelDensityEstimate.

Example
-------
```julia
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
function manifoldProduct( ff::Vector{BallTreeDensity},
                          manif::T;
                          makeCopy::Bool=false,
                          Niter::Int=1,
                          addEntropy::Bool=true,
                          recordLabels::Bool=false,
                          selectedLabels::Vector{Vector{Int}}=Vector{Vector{Int}}()) where {T <: Tuple}
  #
  # check quick exit
  if 1 == length(ff)
    return (makeCopy ? x->deepcopy(x) : x->x)(ff[1])
  end

  ndims = Ndim(ff[1])
  N = Npts(ff[1])


  addopT, diffopT, getManiMu, getManiLam = buildHybridManifoldCallbacks(manif)

  bws = ones(ndims)

  dummy = kde!(rand(ndims,N), bws, addopT, diffopT );

  glbs = KDE.makeEmptyGbGlb();
  glbs.recordChoosen = recordLabels

  pGM, = prodAppxMSGibbsS(dummy, ff,
                          nothing, nothing, Niter=Niter,
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

  bws[:] = getKDEManifoldBandwidths(pGM, manif)
  kde!(pGM, bws, addopT, diffopT)
end

function manifoldProduct( ff::Vector{<:ManifoldKernelDensity},
                          mani::ManifoldsBase.Manifold;
                          kwargs... )
  #
  bels = (x->x.belief).(ff)
  manif = getManifolds(mani)
  manifoldProduct(bels, manif; kwargs...)
end


#
