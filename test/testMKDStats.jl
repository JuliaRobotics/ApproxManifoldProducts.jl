using Test
using ApproxManifoldProducts
using Manifolds

##

@testset "Test basic MKD statistics" begin
##

M = SpecialEuclideanGroup(2; variant = :right)
u0 = ArrayPartition(zeros(2),[1 0; 0 1.0])
ϵ = identity_element(M, typeof(u0))

pts = [exp(M, ϵ, hat(LieAlgebra(M), [0.05*randn(2);0.75*randn()], ArrayPartition)) for i in 1:100]

P = manikde!(M, pts)

#TODO what mean do we want here?
# mean(M, pts, GeodesicInterpolation()) != mean(M, pts)
@test isapprox(M, mean(P), mean(M, pts, GeodesicInterpolation()))
@test isapprox(var(P), var(M, pts))
@test isapprox(std(P), std(M, pts))
@test_broken isapprox(cov(P), cov(M, pts))
@test isapprox(cov(P), cov(M, pts; basis=DefaultOrthogonalBasis()))
@test isapprox(cov(P; basis=DefaultOrthonormalBasis()), cov(M, pts))

##
end
