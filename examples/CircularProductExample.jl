
## Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

using LinearAlgebra
# using DocStringExtensions
using KernelDensityEstimate

using Cairo
using Gadfly, Colors
# using Distributions
# using Random

# using Optim

using ApproxManifoldProducts

using TransformUtils
const TU = TransformUtils



## create two densities

pts1 = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2 = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,) , (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,) , (difftheta,))


pl = plotKDECircular([pc1;pc2])
# pl = plotKDECircular(pc2)



## Quick 1D debug

pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))

x = 2pi*rand(1,1000).-pi
y = pc2(x, false, 1e-3, (addtheta,), (difftheta,))
Gadfly.plot(x=x, y=y, Geom.line)





# pcm.bt



## Calculate the approximate product between the densities


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2,
                          addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
# getBW(pc12)[1,1]
# pc12 = kde!(pGM, [0.1;], (addtheta,), (difftheta,))

pl = plotKDECircular([pc1;pc2; pc12])


## create two densities at +Y



pts1 = TU.wrapRad.(0.1*randn(100).+pi/2.0.+0.5)
pts2 = TU.wrapRad.(0.1*randn(100).+pi/2.0.-0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2,
                          addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
# getBW(pc12)[1,1]
# pc12 = kde!(pGM, [0.1;], addtheta, difftheta)

pl = plotKDECircular([pc1;pc2; pc12])




## Different two



pts1 = TU.wrapRad.(0.1*randn(100).-0.5)
pts2 = TU.wrapRad.(0.1*randn(100).+0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
pl = plotKDECircular([pc1;pc2; pc12])





## another two equally spaced





pts1 = TU.wrapRad.(0.1*randn(100).-pi/2.0)
pts2 = TU.wrapRad.(0.1*randn(100).+pi/2.0)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;]);

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


pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;]);

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


pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;]);

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


pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2; pc3], nothing, nothing, Niter=1, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc123 = kde!_CircularNaiveCV(pGM[:])


pl = plotKDECircular([pc1;pc2;pc3; pc123])




##


Gadfly.push_theme(:default)




pl |> PDF("/tmp/test.pdf", 20cm,15cm)



#
