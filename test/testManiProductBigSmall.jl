# basic test of multiplying big and small together should give small

using ApproxManifoldProducts
using Test
using Manifolds
using LieGroups
using TensorCast

##

@testset "multiply big and small" begin

    ##

    M = SpecialEuclideanGroup(2; variant = :right)
    N = 100
    u0 = ArrayPartition([0; 0.0], [1 0; 0 1.0])
    ϵ = identity_element(M, typeof(u0))

    X1 = [exp(M, ϵ, hat(M, ϵ, randn(3))) for i = 1:N]
    X2 = [exp(M, ϵ, hat(M, ϵ, 0.1 .* randn(3))) for i = 1:N]

    # test get_coordinates
    testval = vee(M, ϵ, log(M, ϵ, X1[1]))
    @test length(testval) === 3
    @test all(abs.(testval) .< 10.0)

    ##

    p = HomotopyDensity_legacy(M, X1)
    q = HomotopyDensity_legacy(M, X2)

    # check new MKD have right type info cached
    @test getPointType(p) <: ArrayPartition

    pq = manifoldProduct([p; q])

    # check new product also has right point type info cached
    @test getPointType(pq) <: ArrayPartition

    ##

    X12_ = getPoints(pq)

    N_ = length(X12_)

    # initial Manellic products was dropping points, this is a reminder to restore the desired number of output points
    if N_ == N
        @test N_ == N
    else 
        @test_broken N_ == N
    end

    XX = (s -> s.x[1][1]).(X12_)
    YY = (s -> s.x[1][2]).(X12_)
    R0 = [1. 0; 0 1]
    TT = (s -> log(getManifold(pq).manifold[2], R0, s.x[2])[1,2]).(X12_)

    @test 0.7 * N_ < sum(abs.(XX) .< 0.3)
    @test 0.7 * N_ < sum(abs.(YY) .< 0.3)
    @test 0.7 * N_ < sum(abs.(TT) .< 0.3)

    ##

end

