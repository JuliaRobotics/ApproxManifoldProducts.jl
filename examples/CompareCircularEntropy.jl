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

# logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
# difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))
# addtheta(wth1::Float64, wth2::Float64) = TU.wrapRad( wth1+wth2 )


# some test points to work with
pts = 0.3*randn(30)
pts = [pts; TU.wrapRad.(0.6*randn(70) .- pi)]
shuffle!(pts)


# p = kde!(pts)

# N = length(pts)
CV  = zeros(100)
global i_global = 0
BW= range(0.05, 1.0, length=100)
for bw in BW
    global i_global
    i_global += 1
    CV[i_global] = manifoldLooCrossValidation(pts, bw, own=true, diffop=difftheta)
end

plot(x=BW, y=CV, Geom.line)



BW[findfirst(CV .== maximum(CV))]

# getBW(kde!(pts))[1,1]

# what we want
# pc = kde!(pts, bw, mani=S1)

# evaluate(pc)

## do the same with

PD = zeros(100)

global i_global = 0
for bw in BW
    global i_global
    i_global += 1
    pd = kde!(pts, [bw;])
    PD[i_global] = KernelDensityEstimate.entropy(pd) # TODO addop, diffop
end





## optimize the KDE bandwidth



# initial testing values
lower = 0.001
upper = 10.0

# pts = 10.0.+randn(100)

minEntropyLOOCV = (bw) -> -manifoldLooCrossValidation(pts, bw, own=true, diffop=difftheta)

# TODO Compare Optim.GoldenSection vs KDE.golden
@time res = optimize(minEntropyLOOCV, lower, upper, GoldenSection(), x_tol=0.001)

bw = res.minimizer


pc = kde!(pts, [bw])

plotKDECircular(pc)

# gg = (x)->pc([x;])[1]
# arr = [gg;]
# pl = plotCircBeliefs(arr)



## Construct circular KDE

pc2 = kde!_CircularNaiveCV(pts)


getBW(pc2)[:,1]


plotKDECircular(pc2)





#
