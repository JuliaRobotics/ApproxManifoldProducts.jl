# first example 2D

using Test
using LinearAlgebra
using ApproxManifoldProducts

using TransformUtils
const TU = TransformUtils

# using Gadfly




#
# logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
# difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))


#assume an unwrapped circle

@testset "Test SO2 wrap around minimization Gaussian means..." begin

@testset "opposing binary..." begin

# expect \mu = 5pi/6
μ1 = 0.0
μ2 = -pi+0.0

Λ1 = 1.0
Λ2 = 1.0

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0), periodicmanifold=wrapRad)[1]
# TU.wrapRad(μ)

N = 1000

BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad)[1] for i in 1:N]
# BUPS[:] .= TU.wrapRad.(BUPS)


@test 0.3*N < sum( 0 .< BUPS .< 3pi/2.0)
@test 0.3*N < sum( -3pi/2 .< BUPS .< 0.0)
@test sum( -0.1 .< BUPS .< 0.1) < 0.01*N

# plot(x=BUPS, Geom.histogram)

end



@testset "binary near wrap..." begin

# expect \mu = 5pi/6
μ1 = -2.6
μ2 = 2.6

Λ1 = 10.0
Λ2 = 10.0

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0), periodicmanifold=wrapRad)[1]
TU.wrapRad(μ)

# doesn't seem to work
# μ = get2DMu(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad)

N = 1000

BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad, initrange=(-pi+0.0,pi+0.0))[1] for i in 1:N]
# BUPS[:] .= TU.wrapRad.(BUPS)


@test 0.3*N < sum( 0 .< BUPS .< 3pi/2.0)
@test 0.3*N < sum( -3pi/2 .< BUPS .< 0.0)
@test 0.01*N < sum( -0.1 .< BUPS .< 0.1) < 0.3*N

# plot(x=BUPS, Geom.histogram)

end

@testset "fanning ternary means..." begin

μ1 = 0.0
μ2 = -4pi/6.0
μ3 = 4pi/6.0

Λ1 = 1.0
Λ2 = 1.0
Λ3 = 1.0

Lambdas = [Λ1; Λ2; Λ3]
mus = [μ1; μ2; μ3]


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad, initrange=(-pi+0.0,pi+0.0))[1]

# TU.wrapRad(μ)

N = 100
BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad, initrange=(-pi+0.0,pi+0.0))[1] for i in 1:N]
# BUPS[:] .= TU.wrapRad.(BUPS)

# plot(x=BUPS, Geom.histogram)

@test 0.20*N < sum( pi/2 .< BUPS .< 3pi/2.0)
@test 0.20*N < sum( -3pi/2 .< BUPS .< -pi/2)
@test 0.20*N < sum( -0.1 .< BUPS .< 0.1)


end

end



@testset "basic linear 2D Rotation find mu and Lambdas Gaussian product test..." begin

# expect \mu = 5pi/6
μ1 = 4pi/6.0
μ2 = -pi+0.0
μ3 = -1.05
# μ3 = -0.5
#
# μ1 = 0.0
# μ2 = 2.0*pi/3.0
# μ3 = -2.0*pi/3.0
#

Λ1 = 1.0
Λ2 = 1.0
Λ3 = 1.0


mus = [μ1; μ2; μ3]
Lambdas = [Λ1; Λ2; Λ3]

difftheta(μ1, 0.0)

difftheta(μ1, -pi+0.0)
difftheta(μ1, pi+0.0)

difftheta(μ1, -1.04)
difftheta(μ1, -1.05)


@test abs(difftheta(μ2, 0.0)) - pi < 1e-8
@test abs(difftheta(μ2, 0.0001) ) - pi < 1e-3
@test abs(difftheta(μ2, -0.0001)) - pi < 1e-3

difftheta(μ2, -1.04)
difftheta(μ2, -1.05)


@test abs(difftheta(μ2, -pi+0.0)) < 1e-8
@test abs(difftheta(μ2, pi+0.0) ) < 1e-8

@test 1e-5 < abs( difftheta(μ1, 5pi/6.0) )
@test 1e-5 < abs( difftheta(μ2, 5pi/6.0) )
@test abs(difftheta(μ1, 5pi/6.0) + difftheta(μ2, 5pi/6.0)) <1e-8

resid2DLinear(0.0, mus, Lambdas, diffop=difftheta)



# @test abs(resid2DLinear(5pi/6.0, mus, Lambdas, diffop=difftheta)) < 1e-10

resid2DLinear(μ1, mus, Lambdas, diffop=difftheta)

resid2DLinear(pi+0.0, mus, Lambdas, diffop=difftheta)
resid2DLinear(-pi, mus, Lambdas, diffop=difftheta)

μ = get2DMu(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad)
Λ = get2DLambda(Lambdas)

# @test abs(μ[1] - 8/5) < 1e-10
#
# @test abs(Λ - 5/4) < 1e-10

end




## plot residual and solution for validation
# using Colors
# using Gadfly
#
# μ1 = 0.0
# μ2 = 2.0*pi/3.0
# μ3 = -2.0*pi/3.0
#
# Λ1 = 1.0
# Λ2 = 1.0
# Λ3 = 1.0
#
# Λ1 = 1.0
# Λ2 = 2.0
# Λ3 = 0.25
#
# mus = [μ1; μ2; μ3]
# Lambdas = [Λ1; Λ2; Λ3]
#
# μ1 = 0.0
# μ2 = -pi+0.0
#
# mus = [μ1; μ2]
# Lambdas = [Λ1; Λ2]
#
# th = range(-pi, stop=pi-1e-5, length=1000)
# obj = map((x) -> resid2DLinear(x, mus, Lambdas, diffop=difftheta), th)
#
# PL = []
#
# # draw resdidual objective for delta angles
# push!( PL, Gadfly.layer(x=th,y=obj, Geom.path, Theme(default_color=colorant"red"))[1])
#
# vals = zeros(1000)
# for i in 1:length(vals)
#   vals[i] = get2DMu(mus, Lambdas, initrange=(-pi+0.0, pi-1e-5), diffop=difftheta, periodicmanifold=wrapRad)[1]
# end
#
# push!(PL, layer(x=vals, Geom.histogram(density=true))[1])
#
# plot(PL...)
#
# 0









#
#
#
#
#
# # Two means 0 and -pi, should have two minima at +- pi/2
# # Three means at 0, +-2pi/3, should have three minima at +-pi/3 and -pi
#
#
#
# μ1 = 0.0
# μ2 = -pi+0.0
# μ2 = 1.0
# mus = [μ1; μ2]
#
# Λ1 = 1.0
# Λ2 = 1.0
# lambdas = [Λ1; Λ2]
#
#
#


# obj .= norm.(obj)
#
#
# X = zeros(length(obj))
# Y = zeros(length(obj))
# global i = 0
# for t in th
#   global i
#   i += 1
#   X[i] = cos(t)*obj[i]
#   Y[i] = sin(t)*obj[i]
# end
#
#
#
# plot(
# Gadfly.layer(x=X, y=Y, Geom.path),
# Gadfly.layer(x=X, y=Y, Geom.path, Theme(default_color=colorant"red")),
# Coord.Cartesian(xmin=-6.0,xmax=6.0,ymin=-6.0,ymax=6.0)
# )
#
# 0
#



#
# # get2DMu(mus, Lambdas, initrange=(2.5+0.0, 2.8))[1, periodicmanifold=wrapRad]
# (x) -> resid2DLinear(x, mus, Lambdas, diffop=difftheta)
# gg2 = (x) -> resid2DLinear(x, mus, Lambdas, diffop=difftheta)
#
#
# using NLsolve
#
# res = zeros(1)
# gg = (res, x) -> solveresid2DLinear(res, x, mus, Lambdas)
# initrange = (-pi+0.0, pi+0.0)
# initr = initrange[2]-initrange[1]
#
# r = NLsolve.nlsolve(gg, [initr*rand()+initrange[1]])
# r = NLsolve.nlsolve(gg, [+1.5])
# r.zero[1]
#
# vals0 = zeros(100)
# vals = zeros(100)
# vals2 = zeros(100)
#
# for i in 1:length(th)
#   res[1] = 0.0
#   # gg(res, th[i])
#   res[1] = resid2DLinear(th[i], mus, Lambdas, diffop=difftheta)
#   vals0[i] = res[1]
#   res[1] = 0.0
#   solveresid2DLinear(res, th[i], mus, Lambdas, diffop=difftheta)
#   vals[i] = res[1]
#   vals2[i] = gg2(th[i])
# end
#
# plot(x=th, y=vals0, Geom.path)
# plot(x=th, y=vals, Geom.path)
# plot(x=th, y=vals2, Geom.path)
#
#
# th = range(-pi, stop=pi, length=100)
#
#
# μ1 = 0.0
# mus = [μ1]
#
# Λ1 = 1.0
# Lambdas = [Λ1]
#
#
# obj = map((x) -> resid2DLinear(x, mus, Lambdas, diffop=difftheta), th)
#
# obj .= norm.(obj)
# # obj2= deepcopy(obj)
# # obj .+= obj2
#
#
# X = zeros(length(obj))
# Y = zeros(length(obj))
# global i = 0
# for t in th
#   global i
#   i += 1
#   X[i] = cos(t)*obj2[i]
#   Y[i] = sin(t)*obj2[i]
# end
#
# plot(x=X, y=Y, Geom.path)
#
# 0


# resid2DLinear(-1.05, mus, Lambdas, diffop=difftheta)



#
# using Gadfly
#
# x = range(-1.1pi, stop=1.1pi, length=1000)
#
# y = difftheta.(0.0,x)
#
# plot(x=x, y=y, Geom.line)
#
# for i in 1:length(x)
#   y[i] = resid2DLinear(x[i], mus, Lambdas, diffop=difftheta)
# end







#
