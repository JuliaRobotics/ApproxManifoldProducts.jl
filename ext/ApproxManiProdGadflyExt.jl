module ApproxManiProdGadflyExt

using Gadfly
using Colors
using Manifolds
using ApproxManifoldProducts: BallTreeDensity, ManifoldKernelDensity

import ApproxManifoldProducts: plotCircBeliefs, plotKDECircular, plotMKD, addtheta, difftheta


include("CircularPlotting.jl")

end