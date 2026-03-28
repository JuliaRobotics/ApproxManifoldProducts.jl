##
# using Revise
using Manifolds
using LieGroups
using ApproxManifoldProducts
using TensorCast
using LinearAlgebra
using Test


##

include(joinpath(@__DIR__, "testutils.jl"))

##


@testset "Partial kernel product with LieGroups.TranslationGroup" begin
##

    d = 2
    M = LieGroups.TranslationGroup(d)
    partial = (1,)

    u = [[1.0; NaN], [0.0; NaN]]
    c = [([1.0; Inf]), ([1.0; Inf])]

    
    @test getManifoldPartial(M, [partial...])[1] isa typeof(LieGroups.TranslationGroup(1))
    @error "expand test for return tuple of getManifoldPartial"
    @test 1 == manifold_dimension(getManifoldPartial(M, [partial...])[1])
 
 
    u_ = ApproxManifoldProducts._mean(M, u; partials=(partial, partial))

    @test isapprox(0.5, u_[1])
    @test isnan(u_[2])

    k1 = ApproxManifoldProducts.MvNormalKernel(u[1], diagm(c[1]); partial=(1,))
    k2 = ApproxManifoldProducts.MvNormalKernel(u[2], diagm(c[2]); partial=(1,))


## calculate extended Gaussian correction term beyond the naive mean, here testing with partials 
    Δμn, Σn = ApproxManifoldProducts.calcProductGaussians_flat(
        M, u, diagm.(c);
        partials = [partial, partial]
    )

    @test isapprox(0.0, Δμn[1])
    @test isapprox(0.0, Δμn[2])
    @test isapprox(0.5, Σn[1, 1])
    @test isapprox(Inf, Σn[2, 2])
    @test isapprox(0.0, Σn[1, 2])
    @test isapprox(0.0, Σn[2, 1])
    


##

    uC = calcProductGaussians(M, [k1; k2])

##

    u_, C_ = mean(uC), cov(uC)
    @test isapprox(u_[1], 0.5)
    @test isnan(u_[2])
    @test isapprox(C_[1,1], 0.5)
    @test isapprox(C_[2,2], Inf)
    @test isapprox(C_[1,2], 0.0)
    @test isapprox(C_[2,1], 0.0)

    @test !isnothing(ApproxManifoldProducts._getprl(uC))
    @test (1,) == ApproxManifoldProducts._getprl(uC)


##
end

@testset "test dim=2 product with one partial/marginal" begin
## basic test one full with one partial

    d = 2
    M = LieGroups.TranslationGroup(d)
    N = 50
    partial = [1;]

    #densities to multiply
    pts1 = [randn(d) for _ = 1:N]
    
    pts2 = [randn(d) for _ = 1:N]
    (x -> (x[2] += NaN)).(pts2) # 100 offset is a decoy to induce errors in case these values are used anywhere
    # TODO, if this were SE2 -> partial=ArrayPartition((2,),(1,)) which replaces legacy [1,3] 


##

    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts2;
        kernel_bw = [1.0; 0.0],
        kernel = AMP.MvNormalKernel,
        partial,
    )

    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(mtree, 1)))
    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelLeaf(mtree, N)))


## check bandwidths of partial belief

    P1 = manikde!(M, pts1)
    P2_ = manikde!(M, pts2; partial)

    @test isapprox( 0.0, getBW(P2_, false)[1][1,2]; atol = 1e-10)
    @test isapprox( Inf, getBW(P2_, false)[1][2,2]; atol = 1e-10)
    @test isapprox( 0.0, getBW(P2_, false)[1][2,1]; atol = 1e-10)


## need tests for partial kernel products

    @test isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(P1.belief,1)))
    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(P2_.belief,1)))

    labels_sampled = [1;1]
    looidx = 1
    tmp_product = ApproxManifoldProducts.calcProductKernelBTLabels(
        M,
        [P1.belief, P2_.belief],
        labels_sampled,
        looidx,  # LOO index
        1:2;
        permute = false,
    )

    @test 2 == length(mean(tmp_product))
    # REMEMBER THIS IS WITH LOOidx=1, so result is just one kernel in product which is also partial
    @test (1,) == ApproxManifoldProducts._getprl(tmp_product)

    # TODO evaluate(M, tmp_product, [0.0])


    eval_at_points =
        [mean(getKernelTree(proposals[O], i, false)) for i in label_pools[O]]
    smw = evaluateDensityAtPoints(M, tmp_product, eval_at_points, true) # TBD: smw = evaluate(tmp_product, )


##


    sl = Vector{Vector{Int}}()
    _labelsChoosen_pp = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)


    P12_ = manifoldProduct(
        [P1; P2_];
        recordLabels = true,
        selectedLabels = sl,
        _labelsChoosen_pp,
        addEntropy = false,
    )

##

    @test !isPartial(P12_)

    @test isapprox(mean(P12_)[1], 0, atol = 1)
    @test_broken isapprox(mean(P12_)[2], 0, atol = 1)

    # @show sl

    P12_

##

    directProductGaussianTestHelper(
        M,
        P1,
        P2_,
        P12_,
        sl,
        pts1,
        pts2,
        N,
    )

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw2 = getBW(P2_, false)[:, 1] .^ 2

        u1 = pts1[sl[sidx][1]]
        u2 = pts2[sl[sidx][2]]

        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])

        @test isapprox(mean(u12)[1], getPoints(P12_)[sidx][1], atol = 0.1)
        @test isapprox(pts1[sl[sidx][1]][2], getPoints(P12_)[sidx][2])
    end

##
end

@testset "test dim=2 product with one full and two similar marginals" begin
## basic test one full with one partial

    N = 50
    d = 2
    M = LieGroups.TranslationGroup(d)

    #densities to multiply
    pts1 = [randn(d) for _ = 1:N]
    P1 = manikde!(M, pts1)

    pts2 = [randn(d) for _ = 1:N]
    (x -> (x[2] += 100)).(pts2)
    P2_ = manikde!(M, pts2; partial = [1;])

    pts3 = [randn(d) for _ = 1:N]
    (x -> (x[2] += 100)).(pts3)
    P3_ = manikde!(M, pts3; partial = [1;])

##

    sl = Vector{Vector{Int}}()

    P123_ = manifoldProduct(
        [P1; P2_; P3_];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P123_)

    @test isapprox(mean(P123_)[1], 0, atol = 1)
    @test_broken isapprox(mean(P123_)[2], 0, atol = 1)

    # @show sl

    P123_

##

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw2 = getBW(P2_, false)[:, 1] .^ 2
        bw3 = getBW(P3_, false)[:, 1] .^ 2

        u1 = pts1[sl[sidx][1]]
        u2 = pts2[sl[sidx][2]]
        u3 = pts3[sl[sidx][3]]

        u123 = calcProductGaussians(M, [u1, u2, u3], [bw1, bw2, bw3])

        @test isapprox(mean(u123)[1], getPoints(P123_)[sidx][1], atol = 0.1)
        @test isapprox(pts1[sl[sidx][1]][2], getPoints(P123_)[sidx][2])
    end

##
end

@testset "test dim=2 product with one full and two different marginals" begin
## random data

    N = 50
    M = LieGroups.TranslationGroup(2)

    pts1 = [randn(2) .- 10.0 for _ = 1:N]
    pts2 = [randn(2) for _ = 1:N]
    pts3 = [randn(2) .+ 10.0 for _ = 1:N]

    # get different marginals

    P1 = marginal(manikde!(M, pts1), [1;])
    P2 = manikde!(M, pts2)
    P3 = marginal(manikde!(M, pts3), [2;])

##

    sl = Vector{Vector{Int}}()
    P = manifoldProduct(
        [P2; P1; P3];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P)

    # @show sl;
    P

## check the results

    pts = getPoints(P)
    @cast pGM[i, j] := pts[j][i]

    @test_broken 0.7 * N < sum(-10 .< pGM[1, :] .< 0)
    @test 0.7 * N < sum(0 .< pGM[2, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw2 = getBW(P2, false)[:, 1] .^ 2
        bw3 = getBW(P3, false)[:, 1] .^ 2

        # full density first
        u2 = pts2[sl[sidx][1]]
        u1 = pts1[sl[sidx][2]]
        u3 = pts3[sl[sidx][3]]

        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        u23 = calcProductGaussians(M, [u2, u3], [bw2, bw3])

        @test isapprox(mean(u12)[1], getPoints(P)[sidx][1])
        @test isapprox(mean(u23)[2], getPoints(P)[sidx][2])
    end

##
end

@testset "product of only one marginal per each of two dimensions" begin
## random data

    N = 50
    M = LieGroups.TranslationGroup(2)

    pts1 = [randn(2) .- 10.0 for _ = 1:N]
    pts3 = [randn(2) .+ 10.0 for _ = 1:N]

    # different marginals

    P1 = marginal(manikde!(M, pts1), [1;])
    P3 = marginal(manikde!(M, pts3), [2;])

## 

    sl = Vector{Vector{Int}}()

    P_ = manifoldProduct(
        [P1; P3];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P_)

    # @show sl

##

    pts = getPoints(P_)
    @cast pGM[i, j] := pts[j][i]

    @test_broken 0.7 * N < sum(-13 .< pGM[1, :] .< -7)
    @test_broken 0.7 * N < sum(7 .< pGM[2, :] .< 13)

## check the selection of labels and resulting Gaussian products are correct

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw3 = getBW(P3, false)[:, 1] .^ 2

        u1 = pts1[sl[sidx][1]]
        u3 = pts3[sl[sidx][2]]

        @test isapprox(u1[1], pts[sidx][1])
        @test isapprox(u3[2], pts[sidx][2])
    end

## 
end

@testset "test dim=2 product of only marginals, two per dimension, 4 factors total" begin
##

    N = 50
    d = 2
    M = LieGroups.TranslationGroup(d)

    pts4 = [randn(d) .- 10.0 for _ = 1:N]
    (x -> x[2] -= 90.0).(pts4)
    pts5 = [randn(d) .+ 10.0 for _ = 1:N]
    (x -> x[1] += 90.0).(pts5)

    P4 = marginal(manikde!(M, pts4), [1;])
    P5 = marginal(manikde!(M, pts5), [d;])

    # test duplication
    pts4_ = [randn(d) .- 10.0 for _ = 1:N]
    (x -> x[2] -= 90.0).(pts4_)
    pts5_ = [randn(d) .+ 10.0 for _ = 1:N]
    (x -> x[1] += 90.0).(pts5_)

    P4_ = marginal(manikde!(M, pts4_), [1;])
    P5_ = marginal(manikde!(M, pts5_), [d;])

##

    sl = Vector{Vector{Int}}()

    P45__ = manifoldProduct(
        [P4; P4_; P5; P5_];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P45__)

    # @show sl;

    P45__

## check the selection of labels and resulting Gaussian products are correct

    # println("getPoints(P45__) = ")
    # getPoints(P45__) .|> println
    # println()

    # sidx = 1
    for sidx = 1:N
        bw1 = getBW(P4, false)[:, 1] .^ 2
        bw2 = getBW(P4_, false)[:, 1] .^ 2
        bw3 = getBW(P5, false)[:, 1] .^ 2
        bw4 = getBW(P5_, false)[:, 1] .^ 2

        u1 = pts4[sl[sidx][1]]
        u2 = pts4_[sl[sidx][2]]
        u3 = pts5[sl[sidx][3]]
        u4 = pts5_[sl[sidx][4]]

        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        u34 = calcProductGaussians(M, [u3, u4], [bw3, bw4])

        # @info "DEBU" string(u1) string(u2) string(u3) string(u4) u12[1], u34[2]
        @test isapprox(mean(u12)[1], getPoints(P45__)[sidx][1])
        @test isapprox(mean(u34)[2], getPoints(P45__)[sidx][2])
    end

##
end

@testset "test dim=3 product with one full and two different marginals" begin
## random data

    d = 3
    N = 50
    M = LieGroups.TranslationGroup(3)

    pts1 = [randn(d) .- 10.0 for _ = 1:N]
    pts2 = [randn(d) for _ = 1:N]
    pts3 = [randn(d) .+ 10.0 for _ = 1:N]

## get different marginals

    P1 = marginal(manikde!(M, pts1), [1;])
    P2 = manikde!(M, pts2)
    P3 = marginal(manikde!(M, pts3), [d;])

##

    # @test_broken false
    # # P = manifoldProduct([P1;P2;P3])
    # @error "manifoldProduct needs complete point type for keyword `oldPoints`, current tests assume no marginal beliefs at front of product array"
    sl = Vector{Vector{Int}}()
    P = manifoldProduct(
        [P2; P1; P3];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P)

    # @show sl;
    P

## check the results

    pts = getPoints(P)
    @cast pGM[i, j] := pts[j][i]

    @test_broken 0.6 * N < sum(-10 .< pGM[1, :] .< 0)
    @test 0.6 * N < sum(0 .< pGM[3, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw2 = getBW(P2, false)[:, 1] .^ 2
        bw3 = getBW(P3, false)[:, 1] .^ 2

        # full density first
        u2 = pts2[sl[sidx][1]]
        u1 = pts1[sl[sidx][2]]
        u3 = pts3[sl[sidx][3]]

        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        u23 = calcProductGaussians(M, [u2, u3], [bw2, bw3])

        @test isapprox(mean(u12)[1], getPoints(P)[sidx][1])
        @test isapprox(u2[2], getPoints(P)[sidx][2])
        @test isapprox(mean(u23)[3], getPoints(P)[sidx][3])
    end

##
end

@testset "test dim=3 product with one full and two different marginals (marginal first in product)" begin
## random data

    d = 3
    N = 50
    M = LieGroups.TranslationGroup(3)

    pts1 = [randn(d) .- 10.0 for _ = 1:N]
    pts2 = [randn(d) for _ = 1:N]
    pts3 = [randn(d) .+ 10.0 for _ = 1:N]

## get different marginals

    P1 = marginal(manikde!(M, pts1), [1;])
    P2 = manikde!(M, pts2)
    P3 = marginal(manikde!(M, pts3), [3;])

##

    sl = Vector{Vector{Int}}()
    P = manifoldProduct(
        [P1; P2; P3];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test !isPartial(P)

    # @show sl;
    P

## check the results

    pts = getPoints(P)
    @cast pGM[i, j] := pts[j][i]

    @test_broken 0.8 * N < sum(-10 .< pGM[1, :] .< 0)
    @test_broken 0.8 * N < sum(0 .< pGM[3, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct

    for sidx = 1:N
        bw1 = getBW(P1, false)[:, 1] .^ 2
        bw2 = getBW(P2, false)[:, 1] .^ 2
        bw3 = getBW(P3, false)[:, 1] .^ 2

        # full density first
        u1 = pts1[sl[sidx][1]]
        u2 = pts2[sl[sidx][2]]
        u3 = pts3[sl[sidx][3]]

        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        u23 = calcProductGaussians(M, [u2, u3], [bw2, bw3])

        @test isapprox(mean(u12)[1], getPoints(P)[sidx][1])
        @test isapprox(u2[2], getPoints(P)[sidx][2])
        @test isapprox(mean(u23)[3], getPoints(P)[sidx][3])
    end

##
end

@testset "test dim=3 product with two different marginals and one open dimension" begin
## random data

    d = 3
    N = 50
    M = LieGroups.TranslationGroup(3)

    pts1 = [randn(d) .- 10.0 for _ = 1:N]
    pts3 = [randn(d) .+ 10.0 for _ = 1:N]

## get different marginals

    P1 = marginal(manikde!(M, pts1), [1;])
    P3 = marginal(manikde!(M, pts3), [d;])

##

    sl = Vector{Vector{Int}}()
    P = manifoldProduct(
        [P1; P3];
        recordLabels = true,
        selectedLabels = sl,
        addEntropy = false,
    )

    @test_broken isPartial(P)
    @test_broken P._partial == [1; 3]

    # @show sl;
    P

## check the results

    pts = getPoints(P, false)
    @cast pGM[i, j] := pts[j][i]

    @test_broken 0.7 * N < sum(-13 .< pGM[1, :] .< -7)
    @test_broken 0.7 * N < sum(7 .< pGM[3, :] .< 13)

## check the selection of labels and resulting Gaussian products are correct

    for sidx = 1:N
        bw1 = getBW(P1)[:, 1] .^ 2
        bw3 = getBW(P3)[:, 1] .^ 2

        # full density first
        u1 = pts1[sl[sidx][1]]
        u3 = pts3[sl[sidx][2]]

        @test isapprox(u1[1], getPoints(P, false)[sidx][1])
        @test isapprox(u3[3], getPoints(P, false)[sidx][3])
    end

##
end

#
