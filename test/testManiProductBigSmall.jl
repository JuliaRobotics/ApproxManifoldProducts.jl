# basic test of multiplying big and small together should give small

using ApproxManifoldProducts
using Test
using Manifolds
using TensorCast

##

@testset "multiply big and small" begin

##

M = SpecialEuclidean(2)
N = 100
u0 = ArrayPartition([0;0.0],[1 0; 0 1.0])
ϵ = identity_element(M, u0)

X1 = [exp(M, ϵ, hat(M, ϵ, randn(3))) for i in 1:N]
X2 = [exp(M, ϵ, hat(M, ϵ, 0.1.*randn(3))) for i in 1:N]

# test get_coordinates
testval = vee(M, ϵ, log(M, ϵ, X1[1]))
@test length(testval) === 3
@test all(abs.(testval) .< 10.0)


##


p = manikde!(M, X1)
q = manikde!(M, X2)

# check new MKD have right type info cached
@test_broken (p._u0 |> typeof) == typeof(u0)

pq = manifoldProduct([p;q], M)

# check new product also has right point type info cached
@test_broken (pq._u0 |> typeof) == typeof(u0)

##

X12_ = getPoints(pq)

X12 = AMP._pointsToMatrixCoords(pq.manifold, X12_)

@test 0.7*N < sum(abs.(X12[1,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[2,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[3,:]) .< 0.3)

##

end
