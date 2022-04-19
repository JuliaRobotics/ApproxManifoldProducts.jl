

using Test
using ApproxManifoldProducts
using Manifolds

##
@testset "test updating of beliefs" begin
##

M = TranslationGroup(2)

pts = [randn(2) for _ in 1:100];
m1 = manikde!(M, pts)


pts = [randn(2) for _ in 1:100];
m2 = manikde!(M, pts; infoPerCoord=[0.3;0.4])

@test 0 < mmd(m1,m2)

AMP._update!(m1, m2)

@test mmd(m1,m2) < 1e-6

@test isapprox(m1.infoPerCoord, m2.infoPerCoord)

##
end

#