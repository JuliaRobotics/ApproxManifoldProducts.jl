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

    
    M_, reprl, partl_cb = getManifoldPartial(M, [partial...])
    @test M_ isa typeof(LieGroups.TranslationGroup(1))
    @test 1 <= manifold_dimension(M_)
    @test_broken reprl isa AbstractVector
    @test partl_cb isa Function

    u_ = ApproxManifoldProducts._mean(M, u; partials=(partial, partial))

    @test isapprox(0.5, u_[1])
    @test isnan(u_[2])

##
    M_, repr, partl_cb = getManifoldPartial(M, partial)

    k1 = ConcentratedGaussianKernel(u[1], diagm(c[1]); partial=(1,), partl_cb)
    k2 = ConcentratedGaussianKernel(u[2], diagm(c[2]); partial=(1,), partl_cb)


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
        kernel = ConcentratedGaussianKernel,
        partial,
    )

    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(mtree, 1)))
    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelLeaf(mtree, N)))


## check bandwidths of partial belief

    P1 = HomotopyDensity_legacy(M, pts1)
    P2_ = HomotopyDensity_legacy(M, pts2; partial)

    # check for normal manikde without partial as control
    @test !isPartial(P1)
    @test 2 == length(mean(ApproxManifoldProducts.getKernelTree(P1, 1)))
    @test !isnan(mean(ApproxManifoldProducts.getKernelTree(P1, 1))[1])
    @test !isnan(mean(ApproxManifoldProducts.getKernelTree(P1, 1))[2])
    @test 2 == length(mean(ApproxManifoldProducts.getKernelLeaf(P1, 1)))
    @test !isnan(mean(ApproxManifoldProducts.getKernelLeaf(P1, 1))[1])
    @test !isnan(mean(ApproxManifoldProducts.getKernelLeaf(P1, 1))[2])

    # check for partial manikde with partial on first dimension, with special care on second coordinate...
    # should give value [x, NaN]...
    @test isPartial(P2_)
    @test 2 == length(mean(ApproxManifoldProducts.getKernelTree(P2_, 1)))
    @test !isnan(mean(ApproxManifoldProducts.getKernelTree(P2_, 1))[1])
    @test isnan(mean(ApproxManifoldProducts.getKernelTree(P2_, 1))[2])
    @test 2 == length(mean(ApproxManifoldProducts.getKernelLeaf(P2_, 1)))
    @test !isnan(mean(ApproxManifoldProducts.getKernelLeaf(P2_, 1))[1])
    @test isnan(mean(ApproxManifoldProducts.getKernelLeaf(P2_, 1))[2])

    # similarly check bandwidths, should have valid values on active coordinates
    @test isapprox( 0.0, getBW(P2_, false)[1][1,2]; atol = 1e-10)
    @test isapprox( Inf, getBW(P2_, false)[1][2,2]; atol = 1e-10)
    @test isapprox( 0.0, getBW(P2_, false)[1][2,1]; atol = 1e-10)


## need tests for partial kernel products

    @test isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(P1,1)))
    @test !isnothing(ApproxManifoldProducts._getprl(ApproxManifoldProducts.getKernelTree(P2_,1)))

    labels_sampled = [1;1]
    looidx = 1
    tmp_product = ApproxManifoldProducts.calcProductKernelBTLabels(
        M,
        [P1, P2_],
        labels_sampled,
        looidx,  # LOO index
        1:2;
        permute = false,
    )

    # mean should be [x, NaN] because looidx=1, so only partial P2_ info for leave-in set
    @test 2 == length(mean(tmp_product))
    @test !isnan(mean(tmp_product)[1])
    @test isnan(mean(tmp_product)[2])
    # REMEMBER THIS IS WITH LOOidx=1, so result is just one kernel in product which is also partial
    @test (1,) == ApproxManifoldProducts._getprl(tmp_product)

    # TODO evaluate(M, tmp_product, [0.0])



##

    sl = Vector{Vector{Int}}()
    _labelsChoosen_pp = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)

    P12_ = manifoldProduct(
        [P1; P2_];
        selectedLabels = sl,
        _labelsChoosen_pp,
    )

##

    @test !isPartial(P12_)
    @test isnothing(ApproxManifoldProducts._getprl(getKernelLeaf(P12_, 1)))

    @test isapprox(mean(P12_)[1], 0, atol = 1)
    @test isapprox(mean(P12_)[2], 0, atol = 1)

    # @show sl

    P12_

##

    partials=[nothing, (1,)]
    hits = 0
    for sidx = 1:Npts(P12_)
        bw1 = getBW(P1, false)[1] 
        bw2 = getBW(P2_, false)[1]

        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        u1 = pts1[sl1_]
        u2 = pts2[sl2_]

        u12, S12, prl = calcProductGaussians(M, [u1, u2], [bw1, bw2]; partials)
        # REMINDER, a similar test for permutation accuracy is in testutils.jl, this is more focused on the partials aspect of the product
        if 1 <= length(findall(≈(u12), getPoints(P12_)))
            hits += 1
        end
    end
    @test 0.8*N < hits

##
end


@testset "product of only one marginal per each of two dimensions" begin
## random data

    N = 50
    M = LieGroups.TranslationGroup(2)

    pts1 = [randn(2) .- 10.0 for _ = 1:N]
    pts3 = [randn(2) .+ 10.0 for _ = 1:N]

    # different marginals

    l1 = [1;]
    l3 = [2;]
    partials = [l1, l3]
    P1_ = HomotopyDensity_legacy(M, pts1)
    P3_ = HomotopyDensity_legacy(M, pts3)
    P1 = marginal(P1_, l1)
    P3 = marginal(P3_, l3)

    @test isPartial(P1)
    @test isPartial(P3)

##

    @test (1,) == ApproxManifoldProducts._getprl(getKernelLeaf(P1, 1))
    @test (1,) == ApproxManifoldProducts._getprl(getKernelTree(P1, 1))

    @test (2,) == ApproxManifoldProducts._getprl(getKernelLeaf(P3, 1))
    @test (2,) == ApproxManifoldProducts._getprl(getKernelTree(P3, 1))

## check marginal kernel products

    p1 = ApproxManifoldProducts.getKernelTree(P1, 2)
    p3 = ApproxManifoldProducts.getKernelTree(P3, 2)

    mvn13 = calcProductGaussians(M, [p1, p3])
    @test isnothing(ApproxManifoldProducts._getprl(mvn13))

    p1 = ApproxManifoldProducts.getKernelTree(P1, 3)
    p3 = ApproxManifoldProducts.getKernelTree(P3, 3)

    mvn13_ = calcProductGaussians(M, [p1, p3])
    @test isnothing(ApproxManifoldProducts._getprl(mvn13_))

    @test !isapprox(mean(mvn13), mean(mvn13_))

##

    sl = Vector{Vector{Int}}()

    @warn "suspect temporary numerical cov issues because product between highly disjoint densities reduce fewer unique product labels -- will get resolved with resample or 'try-harder' after HomotopyDensity refactor."
    P_ = manifoldProduct(
        [P1; P3];
        selectedLabels = sl,
    )

    @test !isPartial(P_)

##

    @test isapprox([-10, 10.0], mean(ApproxManifoldProducts.getKernelTree(P_, 1)); atol = 1.0)
    
    pts = getPoints(P_)
    @cast pGM[i, j] := pts[j][i]

    @test 0.7 * Npts(P_) < sum(-13 .< pGM[1, :] .< -7)
    @test 0.7 * Npts(P_) < sum(7 .< pGM[2, :] .< 13)

## check the selection of labels and resulting Gaussian products are correct

    hits = 0
    for sidx = 1:N
        bw1 = getBW(P1, false)[1]
        bw3 = getBW(P3, false)[1]

        sl1 = [s[1] for s in sl]
        sl3 = [s[2] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        u1 = pts1[sl1_]
        u3 = pts3[sl3_]
        # u1 = pts1[sl[sidx][1]]
        # u3 = pts3[sl[sidx][2]]

        u13, S13, prlc = calcProductGaussians(M, [u1, u3], [bw1, bw3]; partials)
        @test [1,1] == prlc
        if 1 <= length(filter(≈(u13), getPoints(P_)))
            hits += 1
        end

        @test isapprox(-10.0, u1[1]; atol = 4.0)
        @test isapprox(10.0, u3[2]; atol = 4.0)
    end
    @test 0.8*N < hits

## 
end


@testset "test dim=2 product with one full and two similar marginals" begin
## basic test one full with one partial

    N = 50
    d = 2
    M = LieGroups.TranslationGroup(d)

    #densities to multiply
    pts1 = [randn(d) for _ = 1:N]
    P1 = HomotopyDensity_legacy(M, pts1)

    pts2 = [randn(d) for _ = 1:N]
    (x -> (x[2] += 100)).(pts2)
    P2_ = HomotopyDensity_legacy(M, pts2; partial = [1;])

    pts3 = [randn(d) for _ = 1:N]
    (x -> (x[2] += 100)).(pts3)
    P3_ = HomotopyDensity_legacy(M, pts3; partial = [1;])

##

    sl = Vector{Vector{Int}}()

    P123_ = manifoldProduct(
        [P1; P2_; P3_];
        selectedLabels = sl,
    )

    @test !isPartial(P123_)

    @test isapprox(mean(P123_)[1], 0, atol = 1)
    @test isapprox(mean(P123_)[2], 0, atol = 1)

    # @show sl

    P123_

##

    partials=[nothing, (1,), (1,)]

    hits = 0
    for sidx = 1:Npts(P123_)
        bw1 = getBW(P1, false)[1]
        bw2 = getBW(P2_, false)[1]
        bw3 = getBW(P3_, false)[1]

        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl3 = [s[3] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        u1 = pts1[sl1_]
        u2 = pts2[sl2_]
        u3 = pts3[sl3_]
        # u1 = pts1[sl[sidx][1]]
        # u2 = pts2[sl[sidx][2]]
        # u3 = pts3[sl[sidx][3]]

        u123, S123, prl = calcProductGaussians(M, [u1, u2, u3], [bw1, bw2, bw3]; partials)

        if 1 <= length(filter(≈(u123), getPoints(P123_)))
            hits += 1
        end
        # @test isapprox(mean(u123)[1], getPoints(P123_)[sidx][1], atol = 0.1)
        # @test isapprox(pts1[sl[sidx][1]][2], getPoints(P123_)[sidx][2])
    end
    @test 0.8*N < hits

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
    l1 = [1;]
    l3 = [2;]
    partials = [l1, nothing, l3]
    P1 = marginal(HomotopyDensity_legacy(M, pts1), l1)
    P2 = HomotopyDensity_legacy(M, pts2)
    P3 = marginal(HomotopyDensity_legacy(M, pts3), l3)

##

    sl = Vector{Vector{Int}}()
    P = manifoldProduct(
        [P2; P1; P3];
        selectedLabels = sl,
    )

    @test !isPartial(P)

    # @show sl;
    P

## check the results

    pts = getPoints(P)
    @cast pGM[i, j] := pts[j][i]

    @test 0.66 * Npts(P) < sum(-10 .< pGM[1, :] .< 0)
    @test 0.66 * Npts(P) < sum(0 .< pGM[2, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct

    partials=[(1,), nothing, (2,)]
    for sidx = 1:N
        
        bw1 = getBW(P1, false)[1] 
        bw2 = getBW(P2, false)[1] 
        bw3 = getBW(P3, false)[1] 

        # full density first
        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl3 = [s[3] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        u1 = pts1[sl1_]
        u2 = pts2[sl2_]
        u3 = pts3[sl3_]
        # u2 = pts2[sl[sidx][1]]
        # u1 = pts1[sl[sidx][2]]
        # u3 = pts3[sl[sidx][3]]

        u12, S, prl = calcProductGaussians(M, [u1, u2], [bw1, bw2]; partials=[(1,), nothing])
        u23, S, prl = calcProductGaussians(M, [u2, u3], [bw2, bw3]; partials=[nothing, (2,)])

        @test !isapprox(u12[1], u1[1])
        @test !isapprox(u12[2], u1[2])
        @test !isapprox(u12[1], u2[1])
        @test isapprox(u12[2], u2[2])

        @test isapprox(u23[1], u2[1])
        @test !isapprox(u23[2], u2[2])
        @test !isapprox(u23[1], u3[1])
        @test !isapprox(u23[2], u3[2])

        u123, S, prl = calcProductGaussians(M, [u1, u2, u3], [bw1, bw2, bw3]; partials)
        @test isapprox([u12[1]; u23[2]], u123)
        
        # FIXME, maybe something about partial scaling of bandwidths is causing the product 
        #  to not be exactly what the test expects, maybe the test is wrong
        if (1 == length(filter(≈([u12[1]; u23[2]]), getPoints(P)))) && (1 == length(filter(≈(u123), getPoints(P))))
            @test true
        else
            @error "Weak test on product of two different marginals." maxlog=1
            @test_broken false
        end
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

    P4 = marginal(HomotopyDensity_legacy(M, pts4), [1;])
    P5 = marginal(HomotopyDensity_legacy(M, pts5), [d;])

    # test duplication
    pts4_ = [randn(d) .- 10.0 for _ = 1:N]
    (x -> x[2] -= 90.0).(pts4_)
    pts5_ = [randn(d) .+ 10.0 for _ = 1:N]
    (x -> x[1] += 90.0).(pts5_)

    P4_ = marginal(HomotopyDensity_legacy(M, pts4_), [1;])
    P5_ = marginal(HomotopyDensity_legacy(M, pts5_), [d;])

##

    sl = Vector{Vector{Int}}()

    P45__ = manifoldProduct(
        [P4; P4_; P5; P5_];
        selectedLabels = sl,
    )

    @test !isPartial(P45__)

    # @show sl;

    P45__

## check the selection of labels and resulting Gaussian products are correct

    # sidx = 1
    for sidx = 1:N
        bw1 = getBW(P4,  false)[1] 
        bw2 = getBW(P4_, false)[1] 
        bw3 = getBW(P5,  false)[1] 
        bw4 = getBW(P5_, false)[1] 

        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl3 = [s[3] for s in sl]
        sl4 = [s[4] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        sl4_ = sl4[sidx] % N
        sl4_ = sl4_ == 0 ? N : sl4_
        u1 = pts4[sl1_]
        u2 = pts4_[sl2_]
        u3 = pts5[sl3_]
        u4 = pts5_[sl4_]


        u12, S12, prl12 = calcProductGaussians(M, [u1, u2], [bw1, bw2]; partials=[(1,),(1,)])
        u34, S34, prl34 = calcProductGaussians(M, [u3, u4], [bw3, bw4]; partials=[(2,),(2,)])


        @test isapprox(u12[1], -10; atol=3.0)
        @test isapprox(u34[2],  10; atol=3.0)

        @test prl12 == [2, 0]
        @test prl34 == [0, 2]

    end

##
end

@testset "test dim=3 product with one full and two different marginals" begin
## random data

    d = 3
    N = 50
    M = LieGroups.TranslationGroup(d)

    pts1 = [randn(d) .- 10.0 for _ = 1:N]
    pts2 = [randn(d) for _ = 1:N]
    pts3 = [randn(d) .+ 10.0 for _ = 1:N]

## get different marginals

    P1 = marginal(HomotopyDensity_legacy(M, pts1), [1;])
    P2 = HomotopyDensity_legacy(M, pts2)
    P3 = marginal(HomotopyDensity_legacy(M, pts3), [d;])

## weird situation where labels are almost the same, but dims 2, 3 come out the same due to partials

    # NOTICE ORDER SWAP [P2;P1;P3]
    lbls_ = [(78,73,86); (78,74,86)]
    weights = [0.5, 0.5]
    post = ApproxManifoldProducts.calcProductKernelsBTLabels(
        M,
        [P2; P1; P3],
        lbls_,
        false;
        weights,
    )

    # can easily happen (wo resampling) that same labels result in duplication
    @test !isapprox(mean(post[1])[1], mean(post[2])[1]) # different because dim 1 on two labels used on P1
    @test isapprox(mean(post[1])[2], mean(post[2])[2])  # only P2 has info on dim 2
    @test isapprox(mean(post[1])[3], mean(post[2])[3])  # same because reusing the same label on P3 with same P2

##

    sl = Vector{Vector{Int}}()
    # NOTICE ORDER SWAP [P2;P1;P3]
    P = manifoldProduct(
        [P2; P1; P3];
        selectedLabels = sl,
    )

    @test !isPartial(P)

    P

## check the results

    pts = getPoints(P)
    N_ = length(pts)
    @cast pGM[i, j] := pts[j][i]

    @test 0.6 * N_ < sum(-10 .< pGM[1, :] .< 0)
    @test 0.6 * N_ < sum(0 .< pGM[3, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct
    hits = 0
    for sidx = 1:N
        bw1 = getBW(P1, false)[1] 
        bw2 = getBW(P2, false)[1] 
        bw3 = getBW(P3, false)[1] 

        # full density first
        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl3 = [s[3] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        u2 = pts2[sl1_] # notice order swap
        u1 = pts1[sl2_]
        u3 = pts3[sl3_]
        # u2 = pts2[sl[sidx][1]]
        # u1 = pts1[sl[sidx][2]]
        # u3 = pts3[sl[sidx][3]]

        u12, S12, prl12 = calcProductGaussians(M, [u1, u2], [bw1, bw2]; partials=[(1,), nothing])
        u23, S23, prl23 = calcProductGaussians(M, [u2, u3], [bw2, bw3]; partials=[nothing, (d,)])
        u213, S213, prl213 = calcProductGaussians(M, [u2, u1, u3], [bw2, bw1, bw3]; partials=[nothing, (1,), (d,)])

        @test isapprox(u12[2], u23[2])
        if 1 <= length(filter(≈(u213), getPoints(P)))
            hits += 1
        end
        # @test isapprox(u12[1], getPoints(P)[sidx][1])
        # @test isapprox(u2[2], getPoints(P)[sidx][2])
        # @test isapprox(u23[3], getPoints(P)[sidx][3])
    end
    @test_broken 0.8*N < hits

##
end

@testset "test dim=3 product with one full and two different marginals (marginal first in product)" begin
## random data

    d = 3
    N = 50
    M = LieGroups.TranslationGroup(3)

    pts1 = [2*randn(d) .- 10.0 for _ = 1:N]
    pts2 = [2*randn(d) for _ = 1:N]
    pts3 = [2*randn(d) .+ 10.0 for _ = 1:N]

## get different marginals

    P1 = marginal(HomotopyDensity_legacy(M, pts1), [1;])
    P2 = HomotopyDensity_legacy(M, pts2)
    P3 = marginal(HomotopyDensity_legacy(M, pts3), [3;])

##

    sl = Vector{Vector{Int}}()
    @warn "suspect temporary numerical cov issues because product between highly disjoint densities reduce fewer unique product labels -- will get resolved with resample or 'try-harder' after HomotopyDensity refactor."
    P = manifoldProduct(
        [P1; P2; P3];
        selectedLabels = sl,
    )

    @test !isPartial(P)

    # @show sl;
    P

## check the results

    pts = getPoints(P)
    N_ = length(pts)
    @cast pGM[i, j] := pts[j][i]

    @test 0.6 * N_ < sum(-10 .< pGM[1, :] .< 0)
    @test 0.6 * N_ < sum(0 .< pGM[3, :] .< 10)

## check the selection of labels and resulting Gaussian products are correct
    
    hits = 0
    for sidx = 1:N
        bw1 = getBW(P1, false)[1] 
        bw2 = getBW(P2, false)[1] 
        bw3 = getBW(P3, false)[1] 

        # full density first
        sl1 = [s[1] for s in sl]
        sl2 = [s[2] for s in sl]
        sl3 = [s[3] for s in sl]
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        sl3_ = sl3[sidx] % N
        sl3_ = sl3_ == 0 ? N : sl3_
        u1 = pts1[sl1_]
        u2 = pts2[sl2_]
        u3 = pts3[sl3_]

        u12, S12, prl12 = calcProductGaussians(M, [u1, u2], [bw1, bw2]; partials=[(1,), nothing])
        u23, S23, prl23 = calcProductGaussians(M, [u2, u3], [bw2, bw3]; partials=[nothing, (3,)])
        u123, S123, prl123 = calcProductGaussians(M, [u1, u2, u3], [bw1, bw2, bw3]; partials=[(1,), nothing, (3,)])

        @test isapprox(u12[2], u23[2])
        if 1 <= length(filter(≈(u123), getPoints(P)))
            hits += 1
        end
        # @test isapprox(mean(u12)[1], getPoints(P)[sidx][1])
        # @test isapprox(u2[2], getPoints(P)[sidx][2])
        # @test isapprox(mean(u23)[3], getPoints(P)[sidx][3])
    end

    @test_broken 0.8*N < hits
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

    P1 = marginal(HomotopyDensity_legacy(M, pts1), [1;])
    P3 = marginal(HomotopyDensity_legacy(M, pts3), [d;])

##

    try
        sl = Vector{Vector{Int}}()
        P = manifoldProduct(
            [P1; P3];
            selectedLabels = sl,
        )

        @test isPartial(P)
        @test_broken getPartial(P) == [1; 3]

        # @show sl;
        P

    ## check the results

        pts = getPoints(P, false)
        N_ = length(pts)
        @cast pGM[i, j] := pts[j][i]

        @test 0.7 * N_ < sum(-13 .< pGM[1, :] .< -7)
        @test 0.7 * N_ < sum(7 .< pGM[3, :] .< 13)

    ## check the selection of labels and resulting Gaussian products are correct

        pts_ = getPoints(P, false)
        hits = 0
        for sidx = 1:N
            bw1 = getBW(P1)[1] 
            bw3 = getBW(P3)[1] 

            # full density first
            sl1 = [s[1] for s in sl]
            sl3 = [s[3] for s in sl]
            sl1_ = sl1[sidx] % N
            sl1_ = sl1_ == 0 ? N : sl1_
            sl3_ = sl3[sidx] % N
            sl3_ = sl3_ == 0 ? N : sl3_
            u1 = pts1[sl1_]
            u3 = pts3[sl3_]

            if 1 <= length(filter(≈([u1[1]; u3[3]]), (s->s[[1,3]]).(pts_)))
                hits += 1
            end
        end
        @test 0.8*N < hits
    catch e
        @test_broken isa(e, ErrorException) # currently this case throws an error because the product is not supported, but ideally it would just return a partial product with the open dimension
    end
##
end

#
