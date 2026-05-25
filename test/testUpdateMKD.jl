
using Test
using ApproxManifoldProducts
using Manifolds
using LieGroups

##
@testset "test updating of beliefs" begin
##

    M = LieGroups.TranslationGroup(2)

    pts = [randn(2) for _ = 1:100]
    m1 = HomotopyDensity_legacy(M, pts)

    pts = [randn(2) for _ = 1:100]
    m2 = HomotopyDensity_legacy(M, pts; observability = [0.3; 0.4])

    @test 0 < mmd(m1, m2)

    # AMP._update!(m1, m2)

    # @test mmd(m1, m2) < 1e-6

    # @test isapprox(m1.observability, m2.observability)

##
end

#
