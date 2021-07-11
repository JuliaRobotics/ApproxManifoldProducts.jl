
using Test
using ApproxManifoldProducts
using Manifolds

##

@testset "test getManifoldPartial on Euclidean(N)" begin

##

M = Euclidean(3)

@test getManifoldPartial(M, [1;2;3]) == Euclidean(3)

@test getManifoldPartial(M, [2;3]) == Euclidean(2)

##
end

@testset "test getManifoldPartial on Circle()" begin

##

M = Circle()

@test getManifoldPartial(M, [1]) == Circle()

@test_throws ErrorException getManifoldPartial(M, [2;])

# @test 
#  == Euclidean(2)

##
end


@testset "test getManifoldPartial on SpecialEuclidean(2)" begin

##

M = SpecialEuclidean(2)

@test getManifoldPartial(M, [1;2;3]) == SpecialEuclidean(2)

@test getManifoldPartial(M, [1;]) == TranslationGroup(1)
@test getManifoldPartial(M, [2;]) == TranslationGroup(1)
@test getManifoldPartial(M, [1;2]) == TranslationGroup(2)

@test getManifoldPartial(M, [3;]) == SpecialOrthogonal(2)

@test getManifoldPartial(M, [1;3;]) == ProductManifold(TranslationGroup(1), SpecialOrthogonal(2))

##
end

#