# basic test of multiplying big and small together should give small

using ApproxManifoldProducts
using Test

@testset "multiply big and small" begin

N = 100

X1 = randn(3,N)
X2 = 0.1.*randn(3,N)

p = manikde!(X1, (:Euclid, :Euclid, :Circular))
q = manikde!(X2, (:Euclid, :Euclid, :Circular))

pq = manifoldProduct([p;q],(:Euclid,:Euclid,:Circular))

X12 = getPoints(pq)

@test 0.7*N < sum(abs.(X12[1,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[2,:]) .< 0.3)
@test 0.7*N < sum(abs.(X12[3,:]) .< 0.3)

end
