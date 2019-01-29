# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

using DocStringExtensions
using KernelDensityEstimate

include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))


using Gadfly, Colors
using Distributions



# some test points to work with
pts = randn(100)


# p = kde!(pts)

# N = length(pts)
CV  = zeros(100)
global i_global = 0
BW= range(0.1, 1.0, length=100)
for bw in BW
    global i_global
    i_global += 1
    CV[i_global] = manifoldLooCrossValidation(pts, bw, own=true)
end



BW[findfirst(CV .== maximum(CV))]

getBW(kde!(pts))[1,1]



## do the same with

PD = zeros(100)

global i_global = 0
for bw in BW
    global i_global
    i_global += 1
    pd = kde!(pts, [bw;])
    PD[i_global] = KernelDensityEstimate.entropy(pd)
end







##


# plot(y=loo, Geom.point)



plot(
    layer(x=BW, y=CV, Geom.line, Theme(default_color=colorant"red")),
    layer(x=BW, y=-PD.+0.1, Geom.line, Theme(default_color=colorant"blue"))
)


plot(
    layer(x=BW, y=CV+PD, Geom.line, Theme(default_color=colorant"red")),
)




plot(rbf, -5.0, 5.0)


PP = kde!([0.0;], [1.0])

PP = kde!(pts, [1.0])


gg = (x)->PP([x;])[1]

plot(gg, -5.0, 5.0)




0

## characterize computation times

using BenchmarkTools

# function rbftimetest()

val1 = [0.0;]
val2 = [0.0;]

@btime rbf!(val1, 0.9, 0.0, 1.0);

nn = Normal()

# gg = (x::Float64) -> pdf(nn, x)  # even slower with additional allocation

@btime val2[1] = pdf(nn, 0.9);

    # return val1[1], val2[1]
# end
# rbftimetest()



# nn = Normal()
#
# plot(
# layer((x)->kde!(pts)([x;])[1], -5,5),
# layer((x)->pdf(nn,x), -5,5)
# )




## optimize the KDE bandwidth

using Optim


# initial testing values
lower = 0.001
upper = 10.0

pts = 10.0.+randn(100)

minEntropyLOOCV = (bw) -> -manifoldLooCrossValidation(pts, bw)

# TODO Compare Optim.GoldenSection vs KDE.golden
@time res = optimize(minEntropyLOOCV, lower, upper, GoldenSection(), x_tol=0.001)

@time kde!(pts)


res.minimizer


getBW(kde!(pts))[1,1]






#
