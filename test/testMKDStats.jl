using Test
using ApproxManifoldProducts
using Manifolds

@testset "Test basic MKD statistics" begin


M = SpecialEuclidean(2)
u0 = ArrayPartition(zeros(2),[1 0; 0 1.0])
ϵ = identity_element(M, u0)

pts = [exp(M, ϵ, hat(M, ϵ, [0.05*randn(2);0.75*randn()])) for i in 1:100]

P = manikde!(M, pts)

@test isapprox(M, mean(P), mean(M, pts))
@test isapprox(var(P), var(M, pts))
@test isapprox(std(P), std(M, pts))
@test isapprox(cov(P), cov(M, pts))

end
