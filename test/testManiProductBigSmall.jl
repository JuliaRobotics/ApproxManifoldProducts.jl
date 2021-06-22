# basic test of multiplying big and small together should give small

using ApproxManifoldProducts
using Test
using Manifolds
using TensorCast

@testset "multiply big and small" begin

N = 100

X1 = [randn(3) for i in 1:N]
X2 = [0.1.*randn(3) for i in 1:N]

p = manikde!(X1, SpecialEuclidean(2))
q = manikde!(X2, SpecialEuclidean(2))

pq = manifoldProduct([p;q], SpecialEuclidean(2))

X12_ = getPoints(pq)

@cast X12[i,j] := X12_[j][i]

@test 0.7*N < sum(abs.(X12[1,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[2,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[3,:]) .< 0.3)

end
