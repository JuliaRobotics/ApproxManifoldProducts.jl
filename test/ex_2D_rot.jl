# first example 2D

using LinearAlgebra
using ApproxManifoldProducts
using Test

using TransformUtils

const TU = TransformUtils

#assume an unwrapped circle

# expect \mu = 5pi/6
μ1 = 4pi/6.0
μ2 = -pi+0.0
# μ3 = 2.4

Λ1 = 1.0
Λ2 = 1.0
# Λ3 = 5/4.0

# test logmap of SO(2)

# R = Matrix{Float64}(LinearAlgebra.I,2,2)

logmap_SO2(Rl::Matrix{Float64}) = sign(Rl[2,1])*acos(Rl[1,1])
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(@show wth1)'*TU.R(@show wth2))
0.0

@testset "basic linear 2D Rotation find mu and Lambdas Gaussian product test..." begin

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]

difftheta(μ1, 0.0)
difftheta(μ1, pi+0.0)

difftheta(μ2, 0.0)
difftheta(μ2, -pi+0.0)

difftheta(μ1, 5pi/6.0)
difftheta(μ2, 5pi/6.0)

resid2DLinear(0.0, mus, Lambdas, diffop=difftheta)

resid2DLinear(5pi/6.0, mus, Lambdas, diffop=difftheta)

resid2DLinear(μ1, mus, Lambdas, diffop=difftheta)


resid2DLinear(pi+0.0, mus, Lambdas, diffop=difftheta)
resid2DLinear(-pi, mus, Lambdas, diffop=difftheta)

μ = get2DMu(mus, Lambdas, diffop=difftheta)
Λ = get2DLambda(Lambdas)

@test abs(μ[1] - 8/5) < 1e-10

@test abs(Λ - 5/4) < 1e-10

end




using Gadfly

x = range(-2pi, stop=2pi, length=1000);

y = difftheta.(0.0,x);

plot(x=x, y=y, Geom.line)

for i in 1:length(x)
y[i] = resid2DLinear(x[i], mus, Lambdas, diffop=difftheta)
end
