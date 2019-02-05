

# first import plotting
using Cairo
using Gadfly, Colors

# plot features require Gadfly, etc
using ApproxManifoldProducts

# more packages required
using KernelDensityEstimate
using TransformUtils

using RoMEPlotting

# using Distributions
# using Random
# using Optim
# include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))


const KDE = KernelDensityEstimate
const TU = TransformUtils
const AMP = ApproxManifoldProducts



## create two densities


pts1a = 150.0.+10.0*randn(100)
pts2a = 200.0.+10.0*randn(100)

pts1b = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2b = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

pts1 = [pts1a';pts1b']
pts2 = [pts2a';pts2b']

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;1.0], (AMP.addtheta,), (AMP.difftheta,))
pc2 = kde!(pts2, [0.1;1.0], (AMP.addtheta,), (AMP.difftheta,))




# testing Euclidean dimension

pc1a = kde!(pts1a)
pc2a = kde!(pts2a)

dummy = kde!(rand(100),[1.0;]);
pGM, = prodAppxMSGibbsS(dummy, [pc1a; pc2a], nothing, nothing, Niter=1, addop=(+,), diffop=(-,), getMu=(KDE.getEuclidMu,));

pc12a = kde!(pGM)
plotKDE([pc1a;pc2a; pc12a])


# Calculate the approximate product between the hybrid densities

pc1 = kde!(pts1, (+, addtheta), (-,difftheta))
pc2 = kde!(pts2, (+, addtheta), (-,difftheta))

# TODO: make circular KDE
dummy = kde!(rand(2,100),[1.0;]);
pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1, addop=(+,AMP.addtheta), diffop=(-,AMP.difftheta), getMu=(KDE.getEuclidMu, AMP.getCircMu));


pc12a = kde!(pGM[1,:])
getBW(pc12a)[:,1][1]
pc12b = kde!_CircularNaiveCV(pGM[2,:])
getBW(pc12b)[:,1][1]


pc12 = kde!(pGM, [1.4;0.01], (+,addtheta), (-,difftheta))

plotKDE([marginal(pc1,[1;]);marginal(pc2,[1;]); marginal(pc12,[1])])

pl = plotKDECircular([marginal(pc1,[2;]);marginal(pc2,[2;]); marginal(pc12,[2])])




Gadfly.plot((x)->marginal(pc12,[2])([x;])[1], -pi, pi)



#
