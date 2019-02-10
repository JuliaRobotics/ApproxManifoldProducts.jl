
using LinearAlgebra
using Random
using Distributions

# plot features require Gadfly, etc
using ApproxManifoldProducts

# more packages required
using TransformUtils


const LinAlg = LinearAlgebra
const KDE = KernelDensityEstimate
const TU = TransformUtils
const AMP = ApproxManifoldProducts


# temporarily required for experimental versions of on-manifold products
KDE.setForceEvalDirect!(true)



## create two densities



# two densities on a cylinder
p = manikde!(randn(2,100), (:Euclid, :Circular) )

pts2a = 3.0*randn(1,100).+5.0
pts2b = TransformUtils.wrapRad.(0.5*randn(1,100).+pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular), Niter=2)







## better wrap around example


pts1a = 6.0*rand(1,100).-3.0
pts1b = TransformUtils.wrapRad.(0.3*randn(1,100).-0.6*pi)
p = manikde!([pts1a;pts1b], (:Euclid, :Circular) )


pts2a = 6.0*rand(1,100).-3.0
pts2b = TransformUtils.wrapRad.(0.3*randn(1,100).+0.6*pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )


# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular), Niter=3)





## import plotting

using Cairo
using Gadfly, Colors
using KernelDensityEstimatePlotting


## Different plots


plotKDE([p;q; pq], levels=3, c=["red";"green";"magenta"])


plotKDE([p;q; pq], dims=[1],c=["red";"green";"magenta"])

pl = plotKDECircular( [marginal(p, [2;]);marginal(q, [2;]); marginal(pq, [2;])] )

pl = Gadfly.plot(y=pGM[1,:],x=pGM[2,:], Geom.histogram2d(xbincount=50,ybincount=30))



## export the plot to image file

pl |> PDF("/tmp/test.pdf",20cm,15cm)
@async run(`evince /tmp/test.pdf`)
