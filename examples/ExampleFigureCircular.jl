# paper figure for product of circular densities

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







## TOP LEFT two equally spaced





pts1 = TU.wrapRad.(0.1*randn(100).-pi/2.0)
pts2 = TU.wrapRad.(0.1*randn(100).+pi/2.0)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,), (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,), (difftheta,))



dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2, addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])


pltl = plotKDECircular([pc1;pc2; pc12], scale=0.07, c=["green";"blue";"orange"])


pltl.coord = Coord.Cartesian(xmin = -1.6,
xmax = 1.6,
ymin = -1.6,
ymax = 1.6,
aspect_ratio=1.0)

pltl


##

pltl |> PDF("topleft.pdf",8cm,8cm)
pltl |> SVG("topleft.svg",8cm,8cm)
# @async run(`evince topright.pdf`)









## TOP RIGHT

pts1 = TU.wrapRad.(0.1*randn(100).-pi.+0.6)
pts2 = TU.wrapRad.(0.1*randn(100).+pi.-0.6)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;], (addtheta,) , (difftheta,))
pc2 = kde!(pts2, [0.1;], (addtheta,) , (difftheta,))



# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;], (addtheta,), (difftheta,));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=2,
                          addop=(addtheta,), diffop=(difftheta,), getMu=(getCircMu,));


pc12 = kde!_CircularNaiveCV(pGM[:])
getBW(pc12)[1,1]
# pc12 = kde!(pGM, [0.05;], (addtheta,), (difftheta,))

pltr = plotKDECircular([pc1;pc2; pc12], scale=0.07, c=["green";"blue";"orange"])


pltr.coord = Coord.Cartesian(xmin = -1.6,
xmax = 1.6,
ymin = -1.6,
ymax = 1.6,
aspect_ratio=1.0)

pltr


##

pltr |> PDF("topright.pdf",8cm,8cm)
pltr |> SVG("topright.svg",8cm,8cm)
# @async run(`evince topright.pdf`)








## BOTTOM LEFT  handrolically trying product of 3




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


plbl = plotKDECircular([pc1;pc2;pc3; pc123], scale=0.07, c=["green";"blue";"magenta";"orange"])


plbl.coord = Coord.Cartesian(xmin = -1.6,
xmax = 1.6,
ymin = -1.6,
ymax = 1.6,
aspect_ratio=1.0)

plbl

##

plbl |> PDF("bottomleft.pdf",8cm,8cm)
plbl |> SVG("bottomleft.svg",8cm,8cm)
# @async run(`evince topright.pdf`)







## BOTTOM RIGHT product of 3 equally spaced




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

plbr = plotKDECircular([pc1;pc2;pc3; pc123], scale=0.07, c=["green";"blue";"magenta";"orange"])


plbr.coord = Coord.Cartesian(xmin = -1.6,
xmax = 1.6,
ymin = -1.6,
ymax = 1.6,
aspect_ratio=1.0)

plbr

##

plbr |> PDF("bottomright.pdf",8cm,8cm)
plbr |> SVG("bottomright.svg",8cm,8cm)
# @async run(`evince topright.pdf`)




##  Stack


pls = vstack(hstack(pltl,pltr), hstack(plbl,plbr))

##

pls |> PDF("plstack.pdf",12cm,12cm)
pls |> SVG("plstack.svg",12cm,12cm)
