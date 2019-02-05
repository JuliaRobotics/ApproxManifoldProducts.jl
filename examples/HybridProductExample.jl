

# first import plotting
using Cairo
using Gadfly, Colors

# plot features require Gadfly, etc
using ApproxManifoldProducts

# more packages required
using KernelDensityEstimate
using TransformUtils

using RoMEPlotting

using Distributions
# using Random
# using Optim
# include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))


const KDE = KernelDensityEstimate
const TU = TransformUtils
const AMP = ApproxManifoldProducts



KDE.setForceEvalDirect!(true)

## create two densities


pts1a = 180.0.+10.0*randn(100)
pts2a = 200.0.+10.0*randn(100)

pts1b = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2b = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

pts1 = [pts1a';pts1b']
pts2 = [pts2a';pts2b']

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [3.0; 0.1], (+,AMP.addtheta), (-,AMP.difftheta))
pc2 = kde!(pts2, [3.0; 0.1], (+,AMP.addtheta), (-,AMP.difftheta))







dummy = kde!(rand(2,100),[1.0;], (+,AMP.addtheta), (-,AMP.difftheta));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1,
                          addop=(+,AMP.addtheta), diffop=(-,AMP.difftheta), getMu=(KDE.getEuclidMu,AMP.getCircMu));
#



lin1 = getBW(kde!(pGM[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pGM[2,:]))[:,1][1]


pc12 = kde!(pGM, [lin1;cir1], (+,AMP.addtheta), (-,AMP.difftheta))

plotKDE([pc1;pc2; pc12], dims=[1],c=["red";"green";"magenta"])




pl = plotKDECircular([marginal(pc1,[2]);marginal(pc2,[2]); marginal(pc12,[2])])



pl = Gadfly.plot(y=pGM[1,:],x=pGM[2,:], Geom.histogram2d(xbincount=50,ybincount=30))


pl |> PDF("/tmp/test.pdf",20cm,15cm)
@async run(`evince /tmp/test.pdf`)




##





R_ = TU.R(10*pi/180.0)

pts1 = rand(MvNormal([100.0;0.0], R_*[40.0 0.0; 0.0 0.2]*(R_')), 100)
pts1[2,:] = TU.wrapRad.(pts1[2,:])
# plot(x=pts1[1,:],y=pts1[2,:], Geom.histogram2d(xbincount=50,ybincount=30))


R_ = TU.R(-10*pi/180.0)

pts2 = rand(MvNormal([100.0;0.0], R_*[40.0 0.0; 0.0 0.2]*(R_')), 100)
pts2[2,:] = TU.wrapRad.(pts2[2,:])
plot(x=pts2[1,:],y=pts2[2,:], Geom.histogram2d(xbincount=50,ybincount=30))



lin1 = getBW(kde!(pts1[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pts1[2,:]))[:,1][1]



pc1 = kde!(pts1, [lin1; cir1], (+,AMP.addtheta), (-,AMP.difftheta))
pc2 = kde!(pts2, [lin1; cir1], (+,AMP.addtheta), (-,AMP.difftheta))





dummy = kde!(rand(2,100),[1.0;], (+,AMP.addtheta), (-,AMP.difftheta));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1,
                          addop=(+,AMP.addtheta), diffop=(-,AMP.difftheta), getMu=(KDE.getEuclidMu,AMP.getCircMu));
#





lin1 = getBW(kde!(pGM[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pGM[2,:]))[:,1][1]


pc12 = kde!(pGM, [lin1;cir1], (+,AMP.addtheta), (-,AMP.difftheta))

plotKDE([pc1;pc2; pc12], dims=[1],c=["red";"green";"magenta"])




pl = plotKDECircular([marginal(pc1,[2]);marginal(pc2,[2]); marginal(pc12,[2])])





pl = Gadfly.plot(y=pGM[1,:],x=pGM[2,:], Geom.histogram2d(xbincount=50,ybincount=30))



##  Do crazier single and multiple





R_ = TU.R(20*pi/180.0)

pts1 = rand(MvNormal([100.0;0.0], R_*[30.0 0.0; 0.0 0.2]*(R_')), 100)
pts1[2,:] = TU.wrapRad.(pts1[2,:])
# plot(x=pts1[1,:],y=pts1[2,:], Geom.histogram2d(xbincount=50,ybincount=30))



pts2a = [5.0*randn(50).+85; 5.0*randn(50).+115]
pts2b = [TU.wrapRad.(0.1*randn(50).+2.6); TU.wrapRad.(0.1*randn(50).-2.6)]
pts2 = [pts2a';pts2b']

pts2 = pts2[:,randperm(100)]

# plot(x=pts2[1,:],y=pts2[2,:], Geom.histogram2d(xbincount=50,ybincount=30))


lin1 = getBW(kde!(pts1[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pts1[2,:]))[:,1][1]
pc1 = kde!(pts1, [lin1; cir1], (+,AMP.addtheta), (-,AMP.difftheta))

lin2 = getBW(kde!(pts2[1,:]))[:,1][1]
cir2 = getBW(kde!_CircularNaiveCV(pts2[2,:]))[:,1][1]
pc2 = kde!(pts2, [lin2; cir2], (+,AMP.addtheta), (-,AMP.difftheta))





dummy = kde!(rand(2,100),[1.0;], (+,AMP.addtheta), (-,AMP.difftheta));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1,
                          addop=(+,AMP.addtheta), diffop=(-,AMP.difftheta), getMu=(KDE.getEuclidMu,AMP.getCircMu));
#





lin1 = getBW(kde!(pGM[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pGM[2,:]))[:,1][1]


pc12 = kde!(pGM, [lin1;cir1], (+,AMP.addtheta), (-,AMP.difftheta))


pl = Gadfly.plot(x=pGM[1,:],y=pGM[2,:], Geom.histogram2d(xbincount=50,ybincount=30))


# plotKDE([pc1;pc2; pc12], dims=[1],c=["red";"green";"magenta"])





## testing Euclidean dimension separately



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









##



pl |> PDF("/tmp/test.pdf",20cm,15cm)
@async run(`evince /tmp/test.pdf`)





#
