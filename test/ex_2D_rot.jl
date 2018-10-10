# first example 2D

using ApproxManifoldProducts
using Test


Λ1 = 1.0
Λ2 = 0.25
Λ3 = 5/4.0

μ1 = 1.0
μ2 = 4.0
μ3 = 2.4


@testset "basic linear 2D find mu and Lambdas Gaussian product test..." begin

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]

resid2DLinear(0.0, mus, Lambdas)

resid2DLinear(8/5, mus, Lambdas)

μ = get2DMu(mus, Lambdas)
Λ = get2DLambda(Lambdas)

@test abs(μ[1] - 8/5) < 1e-10

@test abs(Λ - 5/4) < 1e-10

end
