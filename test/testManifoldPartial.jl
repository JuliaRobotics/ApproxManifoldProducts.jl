
using Test
using ApproxManifoldProducts
using Manifolds

##

@testset "test getManifoldPartial on Euclidean(N)" begin

##

M = Euclidean(3)

@test getManifoldPartial(M, [1;2;3])[1] == Euclidean(3)
@test getManifoldPartial(M, [2;3])[1] == Euclidean(2)


@test getManifoldPartial(M, [1;2;3], zeros(3))[1] == Euclidean(3)
@test isapprox( getManifoldPartial(M, [1;2;3], zeros(3))[2], zeros(3) )

@test getManifoldPartial(M, [2;3], zeros(3))[1] == Euclidean(2)
@test isapprox( getManifoldPartial(M, [2;3], zeros(3))[2], zeros(2))


##
end

@testset "test getManifoldPartial on Circle()" begin

##

M = Circle()

@test getManifoldPartial(M, [1])[1] == Circle()
@test_throws ErrorException getManifoldPartial(M, [2;])

@test getManifoldPartial(M, [1], [0;])[1] == Circle()
@test getManifoldPartial(M, [1], [0;])[2] == [0]

##
end


@testset "test getManifoldPartial on Rotations(2)" begin

##

M = Rotations(2)

@test getManifoldPartial(M, [1])[1] == Rotations(2)
@test_throws ErrorException getManifoldPartial(M, [2;])

@test getManifoldPartial(M, [1], [1 0; 0 1])[1] == Rotations(2)
@test getManifoldPartial(M, [1], [1 0; 0 1])[2] == [1 0; 0 1]

##
end


@testset "test getManifoldPartial on SpecialEuclidean(2)" begin

##

M = SpecialEuclidean(2)

@test getManifoldPartial(M, [1;2;3])[1] == SpecialEuclidean(2)

@test getManifoldPartial(M, [1;])[1] == TranslationGroup(1)
@test getManifoldPartial(M, [2;])[1] == TranslationGroup(1)
@test getManifoldPartial(M, [1;2])[1] == TranslationGroup(2)

@test getManifoldPartial(M, [3;])[1] == SpecialOrthogonal(2)

@test getManifoldPartial(M, [1;3;])[1] == ProductManifold(TranslationGroup(1), SpecialOrthogonal(2))


repr = ProductRepr([0.0; 0], [1 0; 0 1.0])

@test getManifoldPartial(M, [1;2;3], repr)[1] == SpecialEuclidean(2)
@test getManifoldPartial(M, [1;2;3], repr)[2] == repr

@test getManifoldPartial(M, [1;], repr)[1] == TranslationGroup(1)
@test getManifoldPartial(M, [1;], repr)[2] == [0.0;]

@test getManifoldPartial(M, [2;], repr)[1] == TranslationGroup(1)
@test getManifoldPartial(M, [2;], repr)[2] == [0.0;]

@test getManifoldPartial(M, [1;2], repr)[1] == TranslationGroup(2)
@test getManifoldPartial(M, [1;2], repr)[2] == [0.0;0]

@test getManifoldPartial(M, [3;], repr)[1] == SpecialOrthogonal(2)
@test getManifoldPartial(M, [3;], repr)[2] == repr.parts[2]

@test getManifoldPartial(M, [1;3;], repr)[1] == ProductManifold(TranslationGroup(1), SpecialOrthogonal(2))
r_repr = getManifoldPartial(M, [1;3;], repr)[2]
@test r_repr isa ProductRepr
@test r_repr.parts[1] == [0.0;]
@test r_repr.parts[2] == [1 0; 0 1.0]

##
end


@testset "Reminder, getManifoldPartial on Sphere(2) [TBD]" begin

##

@error "Must fix Sphere(2) partial test"
@test_broken false 

##
end

#