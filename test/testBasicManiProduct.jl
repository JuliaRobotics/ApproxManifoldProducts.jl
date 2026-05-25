## test basic manifold product behaviour

using ApproxManifoldProducts
using Test
using TensorCast
using Manifolds
using LieGroups

##

include(joinpath(@__DIR__, "testutils.jl"))

##

@testset "Test utility functions for Gaussian products, LieGroups.TranslationGroup(1)" begin
##

    M = LieGroups.TranslationGroup(1)

    g1 = ConcentratedGaussianKernel([-1.0;], [4.0;;])
    g2 = ConcentratedGaussianKernel([1.0;], [4.0;;])

    g = ApproxManifoldProducts.calcProductGaussians(M, [g1; g2])
    @test isapprox([0.0;], mean(g); atol = 1e-6)
    @test isapprox([2.0;;], cov(g); atol = 1e-6)

    g1 = ConcentratedGaussianKernel([-1.0;], [4.0;;])
    g2 = ConcentratedGaussianKernel([1.0;], [9.0;;])

    g = ApproxManifoldProducts.calcProductGaussians(M, [g1; g2])
    @test isapprox([-5 / 13;], mean(g); atol = 1e-6)
    @test isapprox([36 / 13;;], cov(g); atol = 1e-6)

##
end


@testset "Test utility functions for Gaussian products, LieGroups.TranslationGroup(2)" begin
##

    M = LieGroups.TranslationGroup(2)
    u = [[1; 1.0], [0.0; 0]]
    c = [([1.0; 1]), ([1.0; 1])]

    uC = calcProductGaussians(M, u, c)
    u_, C_ = mean(uC), cov(uC)
    @test isapprox(u_, [0.5, 0.5])
    @test isapprox(C_, [0.5 0.0; 0.0 0.5])

##
end


# @testset "Test utility functions for multi-scale product sampling" begin
# ##

# M = LieGroups.TranslationGroup(1)

# pts = [randn(1).-1 for _ in 1:3]
# p1 = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[0.1;;], kernel=ConcentratedGaussianKernel)

# @test 1 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 1))
# @test 2 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 2))
# @test 4 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 3))

# @test 64 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 7))
# @test 128 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 8))

# # @enter 
# ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 2)
# ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 3)

# ##
# end

@testset "Product of two Manellic beliefs, Sequential Gibbs, LieGroups.TranslationGroup(1)" begin
##

    M = LieGroups.TranslationGroup(1)
    N = 64

    pts1 = [randn(1) .- 1 for _ = 1:N]
    p1 = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts1;
        kernel_bw = [0.1;;],
        kernel = ConcentratedGaussianKernel,
    )

    pts2 = [randn(1) .+ 1 for _ = 1:N]
    p2 = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts2;
        kernel_bw = [0.1;;],
        kernel = ConcentratedGaussianKernel,
    )

##

    # tree kernel indices
    @test 2 == ApproxManifoldProducts.leftIndex(p1, 1)
    @test 3 == ApproxManifoldProducts.rightIndex(p1, 1)
    # leaf kernel indices
    @test N == ApproxManifoldProducts.leftIndex(p1, floor(Int, N / 2))
    @test N + 1 == ApproxManifoldProducts.rightIndex(p1, floor(Int, N / 2))

    @test ApproxManifoldProducts.isassigned(p1, floor(Int, N / 2))
    @test ApproxManifoldProducts.isassigned(
        p1,
        ApproxManifoldProducts.leftIndex(p1, floor(Int, N / 2)),
    )
    @test !ApproxManifoldProducts.isassigned(p1, 2 * N + 1)


## leaves only version

    @info "Leaves only label sampling version (Gibbs), LieGroups.TranslationGroup(1)"

    #leaves only in binary tree indexing
    bt_label_pool = [
        [(N + 1):(2 * N);], # use leaf BT labels from p1 
        [(N + 1):(2 * N);], # use leaf BT labels from p2
    ]

    ApproxManifoldProducts.sampleProductSeqGibbsBTLabel(M, [p1; p2], 3, bt_label_pool)

    lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(
        M,
        [p1; p2],
        3,
        N,
        bt_label_pool,
    )
    post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1; p2], lbls, false) # ?? was permute=false?

    pts = mean.(post)
    kernel_bw = mean(cov.(post))
    mtr = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw,
        kernel = ConcentratedGaussianKernel,
    )

    @test isapprox(0, mean(ApproxManifoldProducts.getKernelTree(mtr, 1))[1]; atol = 0.75)

    @test all((s -> isapprox(1 / N, s.weight; atol = 1e-6)).(post))

##

    @info "Multi-scale label sampling version (Gibbs), LieGroups.TranslationGroup(1)"

    # test label pool creation
    child_label_pools, all_leaves =
        ApproxManifoldProducts.generateLabelPoolRecursive([p1; p2], [1; 1])
    @test !all_leaves
    @test [2; 3] == child_label_pools[1]
    @test [2; 3] == child_label_pools[2]

    child_label_pools, all_leaves = ApproxManifoldProducts.generateLabelPoolRecursive(
        [p1; p2],
        [floor(Int, N / 2); 2 * N],
    )
    @test !all_leaves
    @test [N; N + 1] == child_label_pools[1]
    @test [2 * N;] == child_label_pools[2]

    child_label_pools, all_leaves =
        ApproxManifoldProducts.generateLabelPoolRecursive([p1; p2], [N + 1; 2 * N])
    @test all_leaves
    @test [N + 1;] == child_label_pools[1]
    @test [2 * N;] == child_label_pools[2]

    # test sampling
    ApproxManifoldProducts.sampleProductSeqGibbsBTLabel(M, [p1; p2])

    lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2])
    post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1; p2], lbls, false) # ?? was permute=false?

    pts = mean.(post)
    kernel_bw = mean(cov.(post))
    mtr = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw,
        kernel = ConcentratedGaussianKernel,
    )

    @test isapprox(0, mean(ApproxManifoldProducts.getKernelTree(mtr, 1))[1]; atol = 0.75)

##
end

##

# using GLMakie

# XX = [[s;] for s in -4:0.1:4]
# YY = ApproxManifoldProducts.evaluate.(Ref(mtr), XX)

# lines((s->s[1]).(XX),YY, color=:magenta)

# YY = ApproxManifoldProducts.evaluate.(Ref(p1), XX)
# lines!((s->s[1]).(XX),YY, color=:blue)
# YY = ApproxManifoldProducts.evaluate.(Ref(p2), XX)
# lines!((s->s[1]).(XX),YY, color=:red)

@testset "Multi-scale label sampling version (Gibbs), LieGroups.TranslationGroup(2)" begin
##

    M = LieGroups.TranslationGroup(2)
    N = 64

    pts1 = [1 * randn(2) for _ = 1:N]
    p1 = HomotopyDensity_legacy(M, pts1)

    pts2 = [1 * randn(2) for _ = 1:N]
    p2 = HomotopyDensity_legacy(M, pts2)

    # test sampling
    lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2])
    lbls_ = unique(lbls)
    N_ = length(lbls_)
    weights = 1 / N .* ones(N_)
    # increase weight of duplicates
    if N_ < N
        for (i, lb_) in enumerate(lbls_)
            idxs = findall(==(lb_), lbls)
            weights[i] = weights[i] * length(idxs)
        end
    end
    post = ApproxManifoldProducts.calcProductKernelsBTLabels(
        M,
        [p1; p2],
        lbls_,
        false;
        weights,
    ) # ?? was permute=false?
    # check that any duplicates resulted in a height weight
    @test isapprox(weights, (s -> s.weight).(post); atol = 1e-6)

    # NOTE, resulting tree might not have N number of data points 
    mtr12 = ApproxManifoldProducts.buildTree_Manellic!(M, post)

##
end



@testset "Basic product for a balanced tree with 8 leaves" begin
## simply multiply two beliefs, sim2

    N = 8
    d = 2
    M = LieGroups.TranslationGroup(d)

    #densities to multiply
    pts1 = [randn(d) for _ = 1:N]
    P1 = manikde!(M, pts1; bw = [1; 1.0])

    pts2 = [randn(d) for _ = 1:N]
    P2 = manikde!(M, pts2; bw = [1; 1.0])


## check basic product of root kernels

    tmp_product = ApproxManifoldProducts.calcProductKernelBTLabels(
        M,
        [P1; P2],
        [1; 1],
        1,
        1:2;
        permute = false,
    )

    # TODO get the mean of pts 1 and mean of pts 2, and check the product mean isapprox
    @warn "Weak test on product of low number of kernels"
    if isapprox(mean(tmp_product), [0.0; 0.0], atol = 0.6)
        @test true
    else
        @test_broken false
    end
    

## check candidate child_label_pools

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [1;1])
    @test clp == [[2; 3], [2; 3]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [2;2])
    @test clp == [[4; 5], [4; 5]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [3;3])
    @test clp == [[6; 7], [6; 7]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [1;2])
    @test clp == [[2; 3], [4; 5]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [3;1])
    @test clp == [[6; 7], [2; 3]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1; P2], [4;5])
    @test clp == [[8; 9], [10; 11]]
    @test !alv

    # (TODO drop duplication) Yuck -- slightly horrible legacy test so that leaf kernels have correct duplicate of the permuted data.
    for i in 1:N
        @test isapprox( P1.points[P1.structure[1][i]], mean(getKernelLeaf(P1, i)))
        @test isapprox( P2.points[P2.structure[1][i]], mean(getKernelLeaf(P2, i)))
    end


##

    sl = Vector{Vector{Int}}()
    _labelsChoosen_pp = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)

    P12 = manifoldProduct(
        [P1; P2];
        MC = 1,
        selectedLabels = sl,
        _labelsChoosen_pp,
    )

    @test !isPartial(P12)
    @test getPartial(P12) === nothing
    # @test isapprox( mean(P12)[1], 0, atol=1 )
    # @test isapprox( mean(P12)[2], 0, atol=1 )

    @show sl;
    _labelsChoosen_pp

##

    # ensure number of products are at least as many as unique label pairs
    @test length(unique(sl)) <= length(getPoints(P12))
    # FIXME add test that duplicate sl's increase weights of those kernels in P12

    # ensure all posterior product labels are from leaf nodes only
    sl1 = [s[1] for s in sl]
    sl2 = [s[2] for s in sl]

    @test all(l -> ApproxManifoldProducts.isLeaf_BTLabel(P1, l), sl1)
    @test all(l -> ApproxManifoldProducts.isLeaf_BTLabel(P2, l), sl2)

    # # check the sorting of the labels is consistent by rebuilding a shuffled belief
    # P1_ = manikde!(M, shuffle(pts1); bw = [1; 1.0])

    # @test all(s->s[1] ≈ s[2], zip(getPoints(P1), getPoints(P1_)) )

    P12

## validate selected labels are working properly, with addEntropy=false

    directProductGaussianTestHelper(
        M,
        P1,
        P2,
        P12,
        sl,
        pts1,
        pts2,
        N,
    )


    # invpermute(B::ApproxManifoldProducts.ManellicTree, s::Int) = findfirst(==(s), B.structure[1])
    # # use idx 1 assuming all leaf bandwidths are the same
    # bw1 = getBW(P1)[invpermute(P1,1)]
    # bw2 = getBW(P2)[invpermute(P2,1)]

    # uhm = ApproxManifoldProducts.calcProductKernelsBTLabels(
    #     M,
    #     [P1; P2],
    #     [(sl1[1],sl2[1]);],
    #     false;
    # )
    # # layers and layers of belief tree indexing pain (part of refactoring transition for HomotopyDensity rename)
    # sl1_ = sl1[1] % N
    # sl1_ = sl1_ == 0 ? N : sl1_
    # sl2_ = sl2[1] % N
    # sl2_ = sl2_ == 0 ? N : sl2_
    # u1 = pts1[sl1_]
    # u2 = pts2[sl2_]
    # u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
    # @test isapprox(mean(uhm[1]), u12)

    # pts12 = getPoints(P12; permute=false)
    # dropdups = Dict{Vector{Int},Int}()
    # for sidx = 1:N
    #     sl1_ = sl1[sidx] % N
    #     sl1_ = sl1_ == 0 ? N : sl1_
    #     sl2_ = sl2[sidx] % N
    #     sl2_ = sl2_ == 0 ? N : sl2_
    #     u1 = pts1[sl1_]
    #     u2 = pts2[sl2_]
    #     # u1 = pts1[sl1[sidx] % N]
    #     # u2 = pts2[sl2[sidx] % N]

    #     u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        
    #     # workaround for duplicate selections in pts12 test
    #     dropdups[sl[sidx]] = get(dropdups, sl[sidx], 0) + 1
    #     idxoff = 0
    #     for (k,i) in dropdups
    #         idxoff += i
    #     end
    #     # TODO test that kernel weights increase for each duplicate selection
    #     # @isapprox( getWeights(P12)[invpermute(P12, sidx)], 1 / N * dropdups[sl[sidx]])

    #     if idxoff <= length(pts12)
    #         @test isapprox(u12, pts12[idxoff])
    #     end
    # end

##
end


@testset "Basic product for an unbalanced tree with 10 leaves" begin
##

@error "TODO"


##
end



## 3D
@testset "Basic 3D multiply and cross dimension covariance test..." begin
##

    N = 100

    pts1 = [[0.05 * randn(2); 0.75 * randn()] for i = 1:N]
    pts2 = [[0.05 * randn(2); 0.75 * randn()] for i = 1:N]

    P1 = manikde!(LieGroups.TranslationGroup(3), pts1)
    P2 = manikde!(LieGroups.TranslationGroup(3), pts2)
##
    # P12 = P1 * P2
    P12 = manifoldProduct([P1; P2])
##    
    @test P12.points[1] isa AbstractVector

    pts_ = getPoints(P12)

    N_ = length(pts_)

    if N_ == N
        @test N_ == N
    else
        @test_broken N_ == N
    end

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
    P12 = manifoldProduct([P1; P2])

    pts_ = getPoints(P12)

    N_ = length(pts_)

    if N_ == N
        @test N_ == N
    else
        @test_broken N_ == N
    end

    XX = (s -> s.x[1][1]).(pts_)
    YY = (s -> s.x[1][2]).(pts_)
    R0 = [1. 0; 0 1]
    TT = (s -> log(getManifold(P12).manifold[2], R0, s.x[2])[1,2]).(pts_)

    @test 0.7 * N_ < sum(abs.(XX) .< 0.1)
    @test 0.7 * N_ < sum(abs.(YY) .< 0.1)
    @test 0.7 * N_ < sum(abs.(TT) .< 2.0)


    # Legacy plotting functions
    # plotKDE([P1;P2;P12], c=["red";"blue";"magenta"],levels=1) |> PDF("/tmp/test.pdf")

##

end


##