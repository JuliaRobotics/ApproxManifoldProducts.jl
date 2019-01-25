# Evaluating S(1) as Manifold for Gaussian Products on manifold

using LinearAlgebra
using ApproxManifoldProducts

using TransformUtils

const TU = TransformUtils

# using Gadfly



logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))


#assume an unwrapped circle


μ1 = # 0.0
μ2 = pi/2.0

Λ1 = 10.0
Λ2 = 10.0

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0), periodicmanifold=wrapRad)[1]
TU.wrapRad(μ)

N = 1000

BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad)[1] for i in 1:N]
# BUPS[:] .= TU.wrapRad.(BUPS)


@test 0.3*N < sum( 0 .< BUPS .< 3pi/2.0)
@test 0.3*N < sum( -3pi/2 .< BUPS .< 0.0)
@test sum( -0.1 .< BUPS .< 0.1) < 0.01*N

# plot(x=BUPS, Geom.histogram)



# @testset "fanning ternary means..." begin
#
# μ1 = 0.0
# μ2 = -4pi/6.0
# μ3 = 4pi/6.0
#
# Λ1 = 1.0
# Λ2 = 1.0
# Λ3 = 1.0
#
# Lambdas = [Λ1; Λ2; Λ3]
# mus = [μ1; μ2; μ3]
#
#
# μ = get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad, initrange=(-pi+0.0,pi+0.0))[1]
#
# # TU.wrapRad(μ)
#
# N = 100
# BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad, initrange=(-pi+0.0,pi+0.0))[1] for i in 1:N]
# # BUPS[:] .= TU.wrapRad.(BUPS)
#
# # plot(x=BUPS, Geom.histogram)
#
# @test 0.20*N < sum( pi/2 .< BUPS .< 3pi/2.0)
# @test 0.20*N < sum( -3pi/2 .< BUPS .< -pi/2)
# @test 0.20*N < sum( -0.1 .< BUPS .< 0.1)
#
#
# end
#
# end
#
#
#
# @testset "basic linear 2D Rotation find mu and Lambdas Gaussian product test..." begin
#
# # expect \mu = 5pi/6
# μ1 = 4pi/6.0
# μ2 = -pi+0.0
# μ3 = -1.05
# # μ3 = -0.5
# #
# # μ1 = 0.0
# # μ2 = 2.0*pi/3.0
# # μ3 = -2.0*pi/3.0
# #
#
# Λ1 = 1.0
# Λ2 = 1.0
# Λ3 = 1.0
#
#
# mus = [μ1; μ2; μ3]
# Lambdas = [Λ1; Λ2; Λ3]
#
# difftheta(μ1, 0.0)
#
# difftheta(μ1, -pi+0.0)
# difftheta(μ1, pi+0.0)
#
# difftheta(μ1, -1.04)
# difftheta(μ1, -1.05)
#
#
# @test abs(difftheta(μ2, 0.0)) - pi < 1e-8
# @test abs(difftheta(μ2, 0.0001) ) - pi < 1e-3
# @test abs(difftheta(μ2, -0.0001)) - pi < 1e-3
#
# difftheta(μ2, -1.04)
# difftheta(μ2, -1.05)
#
#
# @test abs(difftheta(μ2, -pi+0.0)) < 1e-8
# @test abs(difftheta(μ2, pi+0.0) ) < 1e-8
#
# @test 1e-5 < abs( difftheta(μ1, 5pi/6.0) )
# @test 1e-5 < abs( difftheta(μ2, 5pi/6.0) )
# @test abs(difftheta(μ1, 5pi/6.0) + difftheta(μ2, 5pi/6.0)) <1e-8
#
# resid2DLinear(0.0, mus, Lambdas, diffop=difftheta)
#
#
#
# # @test abs(resid2DLinear(5pi/6.0, mus, Lambdas, diffop=difftheta)) < 1e-10
#
# resid2DLinear(μ1, mus, Lambdas, diffop=difftheta)
#
# resid2DLinear(pi+0.0, mus, Lambdas, diffop=difftheta)
# resid2DLinear(-pi, mus, Lambdas, diffop=difftheta)
#
# μ = get2DMu(mus, Lambdas, diffop=difftheta, periodicmanifold=wrapRad)
# Λ = get2DLambda(Lambdas)
#
# # @test abs(μ[1] - 8/5) < 1e-10
# #
# # @test abs(Λ - 5/4) < 1e-10
#
# end
#




#
