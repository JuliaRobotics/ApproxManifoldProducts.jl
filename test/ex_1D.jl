# first example 2D

using ApproxManifoldProducts
using Test

const AMP = ApproxManifoldProducts

Λ1 = 1.0
Λ2 = 0.25
Λ3 = 5/4.0

μ1 = 1.0
μ2 = 4.0
μ3 = 2.4


@testset "basic linear 1D find mu and Lambdas Gaussian product test..." begin

Lambdas = [Λ1; Λ2]
mus = [μ1; μ2]

resid2DLinear(0.0, mus, Lambdas)

resid2DLinear(8/5, mus, Lambdas)

μ = get2DMu(mus, Lambdas)
Λ = get2DLambda(Lambdas)

@test abs(μ[1] - 8/5) < 1e-10

@test abs(Λ - 5/4) < 1e-10

end

## These work, but we are not sure about how to get the scale for product of more than 2 elements

# preS = nullspace(Lambdas*(Lambdas'))
#
# sc = sum(abs.(preS)) / abs(μ1-μ2)
#
# mu = vec(preS / sc)
# Λ = getΛ(Lambdas)
#
#
# u,s,v = svd(Lambdas*(Lambdas'))
#
# s[1,1]*u[:,2]

@testset "test 1D with 3 components on basic Euclidean line..." begin

Lambdas = [Λ1; Λ2; Λ3]
mus = [μ1; μ2; μ3]

μ = get2DMu(mus, Lambdas)
Λ = get2DLambda(Lambdas)

@test abs(μ[1] - 2.0) < 1e-10

@test abs(Λ - 10/4) < 1e-10


end




#
