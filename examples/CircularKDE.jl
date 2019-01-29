# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

using DocStringExtensions
using KernelDensityEstimate

include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))

using Gadfly, Colors
using Distributions
using Random

using TransformUtils
using Optim

const TU = TransformUtils

using ApproxManifoldProducts


# some test points to work with
pts = 0.3*randn(30)
pts = [pts; TU.wrapRad.(0.6*randn(70) .- pi)]
shuffle!(pts)


## Construct circular KDE

pc = kde!_CircularNaiveCV(pts)
getBW(pc)[:,1]

plotKDECircular(pc)



## Create two more densities

pts2 = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts3 = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

pc2 = kde!_CircularNaiveCV(pts2)
pc3 = kde!_CircularNaiveCV(pts3)

plotKDECircular([pc2;pc3])







#
