
using Test
using ApproxManifoldProducts
using Manifolds

##

@testset "test getManifoldPartial on Euclidean(N)" begin

##

M = TranslationGroup(3)

@test getManifoldPartial(M, [1;2;3])[1] == TranslationGroup(3)
@test getManifoldPartial(M, [2;3])[1] == TranslationGroup(2)


@test getManifoldPartial(M, [1;2;3], zeros(3))[1] == TranslationGroup(3)
@test isapprox( getManifoldPartial(M, [1;2;3], zeros(3))[2], zeros(3) )

@test getManifoldPartial(M, [2;3], zeros(3))[1] == TranslationGroup(2)
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

M = Manifolds.Rotations(2)

@test getManifoldPartial(M, [1])[1] == Manifolds.Rotations(2)
@test_throws ErrorException getManifoldPartial(M, [2;])

@test getManifoldPartial(M, [1], [1 0; 0 1])[1] == Manifolds.Rotations(2)
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


repr = ArrayPartition([0.0; 0], [1 0; 0 1.0])

@test getManifoldPartial(M, [1;2;3], repr)[1] == SpecialEuclidean(2)
@test getManifoldPartial(M, [1;2;3], repr)[2] == repr

@test getManifoldPartial(M, [1;], repr)[1] == TranslationGroup(1)
@test getManifoldPartial(M, [1;], repr)[2] == [0.0;]

@test getManifoldPartial(M, [2;], repr)[1] == TranslationGroup(1)
@test getManifoldPartial(M, [2;], repr)[2] == [0.0;]

@test getManifoldPartial(M, [1;2], repr)[1] == TranslationGroup(2)
@test getManifoldPartial(M, [1;2], repr)[2] == [0.0;0]

@test getManifoldPartial(M, [3;], repr)[1] == SpecialOrthogonal(2)
@test getManifoldPartial(M, [3;], repr)[2] == submanifold_component(repr,2)

@test getManifoldPartial(M, [1;3;], repr)[1] == ProductManifold(TranslationGroup(1), SpecialOrthogonal(2))
r_repr = getManifoldPartial(M, [1;3;], repr)[2]
@test r_repr isa ArrayPartition
@test submanifold_component(r_repr,1) == [0.0;]
@test submanifold_component(r_repr,2) == [1 0; 0 1.0]

##
end


@testset "Reminder, getManifoldPartial on Sphere(2) [TBD]" begin

##

@error "Must fix Sphere(2) partial test"
@test_broken false 

##
end


@testset "test getPoints under partial with representation" begin

##

N = 100
M = SpecialEuclidean(2)
u0 = ArrayPartition([0.0; 0], [1 0; 0 1.0])

pts = [exp(M, u0, hat(M, u0, [10 .+ randn(2);randn()])) for i in 1:N]

P = manikde!(M, pts)

P12 = marginal(P, [1;2])

p12 = getPoints(P12)

@test length(p12) == N
@test length(p12[1]) == 2
@test_broken P12.manifold isa TranslationGroup(2)


##
end


@testset "test marginal of marginal (partial) helper" begin
##

M = TranslationGroup(3)
pts = [randn(3) for _ in 1:75]

X = manikde!(M, pts, partial=[1;3])

X_ = marginal(X, [3])

ps3 = getPoints(X_)

for (i,pt) in enumerate(pts)
  @test isapprox(ps3[i][1], pt[3])
end

try
  M = TranslationGroup(4)
  # check the constructor when only a few points are available
  X = manikde(M, pts, partial=[1;3;4])
catch
  @test_broken false
end

##
end


@testset "test replace overloads full and partial/marginal" begin
##

N = 10
M = TranslationGroup(3)
pts0 = [zeros(3) for _ in 1:N]
X0 = manikde!(M, pts0, bw=zeros(3))

pts = [randn(3) for _ in 1:N]
X = manikde!(M, pts)

##

X_ = replace(X0, X)
@test isapprox(X_, X)

##

X = manikde!(M, pts, partial=[1;3])
X_ = replace(X0, X)

# check metadata
@test  isapprox( getBW(X_, false)[[1;3],1], getBW(X,  false)[[1;3],1] )
@test !isapprox( getBW(X_, false)[[1;3],1], getBW(X0, false)[[1;3],1] )

@test  isapprox( X_.infoPerCoord[[1;3]], X.infoPerCoord[[1;3]] )

@test !isPartial(X_)

# check points
x0 = getPoints(X0)
x  = getPoints(X, false)
x_ = getPoints(X_)
for (i,pt) in enumerate(x0)
  # partial of X does not replace 
  @test isapprox(pt[2], x_[i][2])
  @test isapprox(x[i][[1;3]], x_[i][[1;3]])
end


## must also test replace for partial into different partial

pts3 = [randn(3) for _ in 1:N]
X3 = manikde!(M, pts3, partial=[3;])
# and replace partial/marginal values
X__ = replace(X, X3)

@test isPartial(X__)
@test X__._partial == [1;3]

x__ = getPoints(X__, false)
for (i,pt) in enumerate(x)
  @test isapprox(pt[1:2], x__[i][1:2])
  @test isapprox(pts3[i][3], x__[i][3])
end

## union of two partials over all dimensions should drop the partial status

pts12 = [randn(3) for _ in 1:N]
X12 = manikde!(M, pts12, partial=[1;2], infoPerCoord=2*ones(3))

X_np = replace(X12, X3)

@test !isPartial(X_np)
@test isapprox( X_np.infoPerCoord, [2;2;1] )

##
end


#