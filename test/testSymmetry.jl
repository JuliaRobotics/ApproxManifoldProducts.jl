# test for symmetry on distances

using Manifolds
using LieGroups
using Test

import Rotations as _Rot

##
@testset "test symmetry of Manifolds.distance" begin
    ##

    M = LieGroups.TranslationGroup(2)
    a, b = randn(2), randn(2)
    @test isapprox(distance(M, a, b), distance(M, b, a), atol = 1e-5)

    M = SpecialOrthogonalGroup(2)
    a, b = _Rot.RotMatrix(randn()), _Rot.RotMatrix(randn())
    @test isapprox(distance(M, a, b), distance(M, b, a), atol = 1e-5)

    M = SpecialEuclideanGroup(2; variant = :right)
    a = ArrayPartition(randn(2), _Rot.RotMatrix(randn()))
    b = ArrayPartition(randn(2), _Rot.RotMatrix(randn()))
    @test isapprox(distance(M, a, b), distance(M, b, a), atol = 1e-5)

    M = SpecialOrthogonalGroup(3)
    a = _Rot.RotZ(randn()) * _Rot.RotY(randn()) * _Rot.RotX(randn())
    b = _Rot.RotZ(randn()) * _Rot.RotY(randn()) * _Rot.RotX(randn())
    @test isapprox(distance(M, a, b), distance(M, b, a), atol = 1e-5)

    ##
end
