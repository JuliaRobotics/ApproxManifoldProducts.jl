## Test first circular product computations

using DocStringExtensions

include(joinpath(dirname(@__FILE__), "..", "examples", "circularEntropyUtils.jl"))

using KernelDensityEstimate
using Gadfly, Colors
using ApproxManifoldProducts
using Distributions
using Random

using TransformUtils
using Optim

const TU = TransformUtils
const KDE = KernelDensityEstimate

# Define circular manifold

logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))


## test product of just two Gaussian products

# example with correct linear product
mus = [-0.1;0.1]
lams = [100.0; 100.0]

la = KDE.getEuclidLambda(lams)
mu = KDE.getEuclidMu(mus, lams)

# example with incorrect linear product
mus = [-pi+0.1;pi-0.1]
lams = [100.0; 100.0]

la = KDE.getEuclidLambda(lams)
mu = KDE.getEuclidMu(mus, lams)









## Include entropy calculations for bandwidth determination

# some test points to work with
pts = 0.1*randn(30)
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



BW[findfirst(CV .== maximum(CV))]

# getBW(kde!(pts))[1,1]



## do the same with

PD = zeros(100)

global i_global = 0
for bw in BW
    global i_global
    i_global += 1
    pd = kde!(pts, [bw;])
    PD[i_global] = KernelDensityEstimate.entropy(pd)
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


gg = (x)->pc([x;])[1]

arr = [gg;]


pl = plotCircBeliefs(arr)




#
