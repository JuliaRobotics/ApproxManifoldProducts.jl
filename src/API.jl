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
function manifoldProduct(ff::Vector{BallTreeDensity},
                         manif::T;
                         Niter=1  )::BallTreeDensity where {T <: Tuple}
  #

  ndims = Ndim(ff[1])
  N = Npts(ff[1])

  addopT, diffopT, getManiMu, getManiLam = buildHybridManifoldCallbacks(manif)

  bws = ones(ndims)

  dummy = kde!(rand(ndims,N), bws, addopT, diffopT );

  pGM, = prodAppxMSGibbsS(dummy, ff,
                          nothing, nothing, Niter=1,
                          addop=addopT,
                          diffop=diffopT,
                          getMu=getManiMu  );
  #

  bws[:] = getKDEManifoldBandwidths(pGM, manif)
  kde!(pGM, bws, addopT, diffopT)
end




#
