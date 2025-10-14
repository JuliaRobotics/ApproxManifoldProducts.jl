# test manifold conventions

using Test
using LieGroups
using RecursiveArrayTools
import Rotations as _Rot

##
@testset "test MF.jl conventions" begin
    ##
    # TODO why this test?
    # This test used to test deprecated SpecialEuclidean(2; vectors = HybridTangentRepresentation())
    # this is the equivelent to make it work in LieGroups.jl
    M = TranslationGroup(2) × SpecialOrthogonalGroup(2)
    G = SpecialEuclideanGroup(2; variant = :right)
    e0 = identity_element(M, ArrayPartition)

    # body to body next
    bTb_ = ArrayPartition([10.0; 0], _Rot.RotMatrix(pi / 2))

    # drive in a clockwise square from the origin
    wTb0 = ArrayPartition([100.0; 0], _Rot.RotMatrix(0.0))
    wTb1 = compose(G, wTb0, bTb_)    # right
    wTb2 = compose(G, wTb1, bTb_)    # top right
    wTb3 = compose(G, wTb2, bTb_)    # top
    wTb4 = compose(G, wTb3, bTb_)    # origin

    wCb0 = vee(M, e0, log(M, e0, wTb0))
    wCb1 = vee(M, e0, log(M, e0, wTb1))
    wCb2 = vee(M, e0, log(M, e0, wTb2))
    wCb3 = vee(M, e0, log(M, e0, wTb3))
    wCb4 = vee(M, e0, log(M, e0, wTb4))

    ##

    # check the favorable result
    @test isapprox([100, 0.0, 0], wCb0; atol = 1e-6)
    @test isapprox([110, 0.0, pi / 2], wCb1; atol = 1e-6)
    @test isapprox([110.0, 10, pi], wCb2; atol = 1e-6) ||
          isapprox([110, 10, -pi], wCb2; atol = 1e-6)
    @test isapprox([100.0, 10, -pi / 2], wCb3; atol = 1e-6)
    @test isapprox([100, 0, 0.0], wCb4; atol = 1e-6)

    ## check that the inverse breaks

    # Use opposite convention from above to show it is wrong
    wTb0 = ArrayPartition([100.0; 0], _Rot.RotMatrix(0.0))
    wTb1 = compose(G, bTb_, wTb0)    # right
    wTb2 = compose(G, bTb_, wTb1)    # top right
    wTb3 = compose(G, bTb_, wTb2)    # top
    wTb4 = compose(G, bTb_, wTb3)    # origin

    wCb0 = vee(M, e0, log(M, e0, wTb0))
    wCb1 = vee(M, e0, log(M, e0, wTb1))
    wCb2 = vee(M, e0, log(M, e0, wTb2))
    wCb3 = vee(M, e0, log(M, e0, wTb3))
    wCb4 = vee(M, e0, log(M, e0, wTb4))

    # check the negative result
    @test isapprox([100, 0.0, 0], wCb0; atol = 1e-6)
    @test !isapprox([110, 0.0, pi / 2], wCb1; atol = 1e-6)
    @test !(
        isapprox([110.0, 10, pi], wCb2; atol = 1e-6) ||
        isapprox([110, 10, -pi], wCb2; atol = 1e-6)
    )
    @test !isapprox([100.0, 10, -pi / 2], wCb3; atol = 1e-6)
    @test isapprox([100, 0, 0.0], wCb4; atol = 1e-6)

    ##
end
