# test basic manifold product behaviour

using ApproxManifoldProducts
# using KernelDensityEstimate
using Test
using TensorCast
using Manifolds
using LieGroups

## 3D

@testset "Basic 3D multiply and cross dimension covariance test..." begin

    ##

    N = 100

    pts1 = [[0.05 * randn(2); 0.75 * randn()] for i = 1:N]
    pts2 = [[0.05 * randn(2); 0.75 * randn()] for i = 1:N]

    P1 = manikde!(TranslationGroup(3), pts1)
    P2 = manikde!(TranslationGroup(3), pts2)

    # P12 = P1 * P2
    P12 = manifoldProduct([P1; P2], TranslationGroup(3); legacy = false)

    @test typeof(P12._u0) <: Vector{Float64}

    pts_ = getPoints(P12)

    N_ = length(pts_)

    if N_ == N
        @test N_ == N
    else
        @test_broken N_ == N
    end

    # pts = AMP._pointsToMatrixCoords(P12.manifold, pts_)

    @test 0.8 * N_ < sum(abs.((s->s[1] < 0.1).(pts_)))
    @test 0.8 * N_ < sum(abs.((s->s[2] < 0.1).(pts_)))
    @test 0.8 * N_ < sum(abs.((s->s[3] < 2.0).(pts_)))

    ##

    M = SpecialEuclideanGroup(2; variant = :right)
    u0 = ArrayPartition(zeros(2), [1 0; 0 1.0])
    ϵ = identity_element(M, typeof(u0))

    pts1 = [exp(M, ϵ, hat(M, ϵ, [0.05 * randn(2); 0.75 * randn()])) for i = 1:N]
    pts2 = [exp(M, ϵ, hat(M, ϵ, [0.05 * randn(2); 0.75 * randn()])) for i = 1:N]

    P1 = manikde!(M, pts1)
    P2 = manikde!(M, pts2)

    # P12 = P1 * P2
    P12 = manifoldProduct([P1; P2], M; legacy = false)

    pts_ = getPoints(P12)

    N_ = length(pts_)

    if N_ == N
        @test N_ == N
    else
        @test_broken N_ == N
    end

    # pts = AMP._pointsToMatrixCoords(P12.manifold, pts_)

    XX = (s -> s.x[1][1]).(pts_)
    YY = (s -> s.x[1][2]).(pts_)
    R0 = [1. 0; 0 1]
    TT = (s -> log(P12.manifold.manifold[2], R0, s.x[2])[1,2]).(pts_)

    @test 0.7 * N_ < sum(abs.(XX) .< 0.1)
    @test 0.7 * N_ < sum(abs.(YY) .< 0.1)
    @test 0.7 * N_ < sum(abs.(TT) .< 2.0)


    # Legacy plotting functions
    # plotKDE([P1;P2;P12], c=["red";"blue";"magenta"],levels=1) |> PDF("/tmp/test.pdf")

    ##

end

#
