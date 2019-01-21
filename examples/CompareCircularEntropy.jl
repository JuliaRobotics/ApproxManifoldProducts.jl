# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold


include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))


using KernelDensityEstimate
using Gadfly, Colors
using Distributions

using TransformUtils

const TU = TransformUtils


logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))



# some test points to work with
pts = randn(100)
pts = TU.wrapRad.(randn(100) .- pi)



# p = kde!(pts)

# N = length(pts)
CV  = zeros(100)
global i_global = 0
BW= range(0.05, 1.0, length=100)
for bw in BW
    global i_global
    i_global += 1
    CV[i_global] = looCrossValidation(pts, bw, own=true, diffop=difftheta)
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

using Optim


# initial testing values
lower = 0.001
upper = 10.0

# pts = 10.0.+randn(100)

minEntropyLOOCV = (bw) -> -looCrossValidation(pts, bw, own=true, diffop=difftheta)

# TODO Compare Optim.GoldenSection vs KDE.golden
@time res = optimize(minEntropyLOOCV, lower, upper, GoldenSection(), x_tol=0.001)

bw = res.minimizer


pc = kde!(pts, [bw])


gg = (x)->pc([x;])[1]

arr = [gg;]


pl = plotCircBeliefs(arr)




#
