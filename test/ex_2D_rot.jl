# first example 2D

using Test
using LinearAlgebra
using ApproxManifoldProducts

using TransformUtils

const TU = TransformUtils

# using Gadfly

import ApproxManifoldProducts: *

struct SO2Manifold <: Manifolds
end


# should not be defined in AMP, since we want IIF indepent of manifolds
function *(PP::Vector{MB{SO2Manifold,B}}) where B
  @info "taking manifold product of $(length(PP)) terms"
  @warn "SO2Manifold: work in progress"
end

mbr1 = ManifoldBelief(SO2Manifold, 0.0)
mbr2 = ManifoldBelief(SO2Manifold, 0.0)

*([mbr1;mbr2])


logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))


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


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0))[1]
TU.wrapRad(μ)

N = 1000

BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta)[1] for i in 1:N]
BUPS[:] .= TU.wrapRad.(BUPS)


@test 0.3*N < sum( 0 .< BUPS .< 3pi/2.0)
@test 0.3*N < sum( -3pi/2 .< BUPS .< 0.0)
@test sum( -0.1 .< BUPS .< 0.1) < 0.01*N

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


μ = get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0))[1]

TU.wrapRad(μ)

BUPS = Float64[get2DMuMin(mus, Lambdas, diffop=difftheta, initrange=(-pi+0.0,pi+0.0))[1] for i in 1:N]
BUPS[:] .= TU.wrapRad.(BUPS)

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

Λ1 = 1.0
Λ2 = 1.0
Λ3 = 1.0


Lambdas = [Λ1; Λ2; Λ3]
mus = [μ1; μ2; μ3]

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

@test abs(resid2DLinear(5pi/6.0, mus, Lambdas, diffop=difftheta)) < 1e-10

resid2DLinear(μ1, mus, Lambdas, diffop=difftheta)

resid2DLinear(pi+0.0, mus, Lambdas, diffop=difftheta)
resid2DLinear(-pi, mus, Lambdas, diffop=difftheta)

μ = get2DMu(mus, Lambdas, diffop=difftheta)
Λ = get2DLambda(Lambdas)

@test abs(μ[1] - 8/5) < 1e-10

@test abs(Λ - 5/4) < 1e-10

end





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
