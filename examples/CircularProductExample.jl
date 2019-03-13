
## Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

using LinearAlgebra
# using DocStringExtensions
using KernelDensityEstimate

using Cairo
using Gadfly, Colors
using Random

using ApproxManifoldProducts

using TransformUtils
const TU = TransformUtils



setForceEvalDirect!(true)

## single Bimodal density



pts = [0.1*randn(50); TU.wrapRad.(0.1*randn(50).+pi)]
shuffle!(pts)

pp = kde!(pts, [0.05], (addtheta,), (difftheta,))

plotKDECircular(pp)

##

pp = kde!(pts, [0.05], (addtheta,), (difftheta,))

x = 2pi*rand(1,1000).-pi
y = pp(x)

Gadfly.plot(x=x, y=y, Geom.line)


## create two densities

pts1 = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2 = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,) , (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,) , (difftheta,))


pl = plotKDECircular([pc1;pc2])



## Calculate the approximate product between the densities


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2,
                          addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
getBW(pc12)[1,1]
# pc12 = kde!(pGM, [0.05;], (addtheta,), (difftheta,))

pl = plotKDECircular([pc1;pc2; pc12], scale=0.07, c=["cyan";"blue";"red"])


# x = reshape(collect(range(-pi, stop=pi, length=1000)),1,:)
# y = pc12(x, false, 1e-3, (addtheta,), (difftheta,))
# Gadfly.plot(x=x, y=y, Geom.line)

# Gadfly.push_theme(:default)
pl |> PDF("/tmp/test.pdf",10cm,10cm)
@async run(`evince /tmp/test.pdf`)

pl.coord = Coord.Cartesian(xmin = -1.8,
xmax = 1.8,
ymin = -1.8,
ymax = 1.8,
aspect_ratio=1.0)

pl.theme
pl.theme.line_width = 2.0mm

pl
## create two densities at +Y

N = 100

pts1 = TU.wrapRad.(0.1*randn(N).+pi/2.0.+0.5)
pts2 = TU.wrapRad.(0.1*randn(N).+pi/2.0.-0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(N),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2,
                          addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
# getBW(pc12)[1,1]
# pc12 = kde!(pGM, [0.1;], addtheta, difftheta)

pl = plotKDECircular([pc1;pc2; pc12])




## Different two


N=100

pts1 = TU.wrapRad.(0.1*randn(N).-0.5)
pts2 = TU.wrapRad.(0.1*randn(N).+0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(N),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


# pc12 = kde!_CircularNaiveCV(pGM[:])
# getBW(pc12)[:,1][1]

pc12 = kde!(pGM,[0.05], (addtheta,), (difftheta,));

pl = plotKDECircular([pc1;pc2; pc12])




## another two equally spaced





pts1 = TU.wrapRad.(0.1*randn(100).-pi/2.0)
pts2 = TU.wrapRad.(0.1*randn(100).+pi/2.0)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
pl = plotKDECircular([pc1;pc2; pc12])






## handrolically trying product of 3




pts1 = TU.wrapRad.(0.1*randn(100))
pts2 = TU.wrapRad.(0.1*randn(100).-2pi/3.0.+1.0)
pts3 = TU.wrapRad.(0.1*randn(100).+2pi/3.0)


# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))
pc3 = kde!(pts3, [0.1;], (addtheta,), (difftheta,))


# pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2; pc3], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc123 = kde!_CircularNaiveCV(pGM[:])


pl = plotKDECircular([pc1;pc2;pc3; pc123])




## product of 3




pts1 = TU.wrapRad.(0.1*randn(100))
pts2 = TU.wrapRad.(0.1*randn(100).-2pi/3.0.+1.0)
pts3 = TU.wrapRad.(0.1*randn(100).+2pi/3.0.-1.0)


# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))
pc3 = kde!(pts3, [0.1;], (addtheta,), (difftheta,))


# pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2; pc3], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc123 = kde!_CircularNaiveCV(pGM[:])


pl = plotKDECircular([pc1;pc2;pc3; pc123])






## product of 3




pts1 = TU.wrapRad.(0.1*randn(100))
pts2 = TU.wrapRad.(0.1*randn(100).-2pi/3.0)
pts3 = TU.wrapRad.(0.1*randn(100).+2pi/3.0)


# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))
pc3 = kde!(pts3, [0.1;], (addtheta,), (difftheta,))


# pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2; pc3], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc123 = kde!_CircularNaiveCV(pGM[:])


pl = plotKDECircular([pc1;pc2;pc3; pc123])




##


Gadfly.push_theme(:default)




pl |> PDF("/tmp/test.pdf", 20cm,15cm)



#
