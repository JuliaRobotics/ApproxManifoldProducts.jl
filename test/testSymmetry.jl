# test for symmetry on distances

using Manifolds
using Test

import Rotations as _Rot

##
@testset "test symmetry of Manifolds.distance" begin
##

M = TranslationGroup(2)
a,b = randn(2), randn(2)
@test isapprox( distance(M, a, b), distance(M, b, a), atol = 1e-5)


M = SpecialOrthogonal(2)
a,b = _Rot.RotMatrix(randn()), _Rot.RotMatrix(randn())
@test isapprox( distance(M, a, b), distance(M, b, a), atol = 1e-5)


M = SpecialEuclidean(2)
a = ArrayPartition(randn(2),_Rot.RotMatrix(randn()))
b = ArrayPartition(randn(2),_Rot.RotMatrix(randn()))
@test isapprox( distance(M, a, b), distance(M, b, a), atol = 1e-5)


M = SpecialOrthogonal(3)
a = _Rot.RotZ(randn())*_Rot.RotY(randn())*_Rot.RotX(randn())
b = _Rot.RotZ(randn())*_Rot.RotY(randn())*_Rot.RotX(randn())
@test isapprox( distance(M, a, b), distance(M, b, a), atol = 1e-5)

##
end