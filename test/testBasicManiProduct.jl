## test basic manifold product behaviour

using ApproxManifoldProducts
using Test
using TensorCast
using Manifolds
using LieGroups


##

@testset "Test utility functions for Gaussian products, LieGroups.TranslationGroup(1)" begin
##

    M = LieGroups.TranslationGroup(1)

    g1 = ApproxManifoldProducts.MvNormalKernel([-1.0;], [4.0;;])
    g2 = ApproxManifoldProducts.MvNormalKernel([1.0;], [4.0;;])

    g = ApproxManifoldProducts.calcProductGaussians(M, [g1; g2])
    @test isapprox([0.0;], mean(g); atol = 1e-6)
    @test isapprox([2.0;;], cov(g); atol = 1e-6)

    g1 = ApproxManifoldProducts.MvNormalKernel([-1.0;], [4.0;;])
    g2 = ApproxManifoldProducts.MvNormalKernel([1.0;], [9.0;;])

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
# p1 = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

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
        kernel = ApproxManifoldProducts.MvNormalKernel,
    )

    pts2 = [randn(1) .+ 1 for _ = 1:N]
    p2 = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts2;
        kernel_bw = [0.1;;],
        kernel = ApproxManifoldProducts.MvNormalKernel,
    )

##

    # tree kernel indices
    @test 2 == ApproxManifoldProducts.leftIndex(p1, 1)
    @test 3 == ApproxManifoldProducts.rightIndex(p1, 1)
    # leaf kernel indices
    @test N + 1 == ApproxManifoldProducts.leftIndex(p1, floor(Int, N / 2))
    @test N + 2 == ApproxManifoldProducts.rightIndex(p1, floor(Int, N / 2))

    @test ApproxManifoldProducts.exists_BTLabel(p1, floor(Int, N / 2))
    @test ApproxManifoldProducts.exists_BTLabel(
        p1,
        ApproxManifoldProducts.leftIndex(p1, floor(Int, N / 2)),
    )
    @test !ApproxManifoldProducts.exists_BTLabel(p1, 2 * N + 1)


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
        kernel = ApproxManifoldProducts.MvNormalKernel,
    )

    @test isapprox(0, mean(ApproxManifoldProducts.getKernelTree(mtr, 1))[1]; atol = 0.75)

    @test all((s -> isapprox(1 / N, s.shim.weight; atol = 1e-6)).(post))

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
    @test [N + 1; N + 2] == child_label_pools[1]
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
        kernel = ApproxManifoldProducts.MvNormalKernel,
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
    p1 = ApproxManifoldProducts.manikde!(M, pts1)

    pts2 = [1 * randn(2) for _ = 1:N]
    p2 = ApproxManifoldProducts.manikde!(M, pts2)

    # test sampling
    lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1.belief; p2.belief])
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
        [p1.belief; p2.belief],
        lbls_,
        false;
        weights,
    ) # ?? was permute=false?
    # check that any duplicates resulted in a height weight
    @test isapprox(weights, (s -> s.shim.weight).(post); atol = 1e-6)

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
        [P1.belief; P2.belief],
        [1; 1],
        1,
        1:2;
        permute = false,
    )

    # TODO get the mean of pts 1 and mean of pts 2, and check the product mean isapprox
    @test_broken isapprox(mean(tmp_product), [0.0, 0.0], atol = 0.1)

## check candidate child_label_pools

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [1;1])
    @test clp == [[2; 3], [2; 3]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [2;2])
    @test clp == [[4; 5], [4; 5]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [3;3])
    @test clp == [[6; 7], [6; 7]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [1;2])
    @test clp == [[2; 3], [4; 5]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [3;1])
    @test clp == [[6; 7], [2; 3]]
    @test !alv

    clp, alv = ApproxManifoldProducts.generateLabelPoolRecursive([P1.belief; P2.belief], [4;5])
    @test clp == [[9; 10], [11; 12]]
    @test !alv

    # (TODO drop duplication) Yuck -- slightly horrible legacy test so that leaf kernels have correct duplicate of the permuted data.
    for i in 1:N
        @test isapprox( P1.belief.data[P1.belief.permute[i]], mean(P1.belief.leaf_kernels[i]))
        @test isapprox( P2.belief.data[P2.belief.permute[i]], mean(P2.belief.leaf_kernels[i]))
    end


##

    sl = Vector{Vector{Int}}()
    _labelsChoosen_pp = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)

    P12 = manifoldProduct(
        [P1; P2];
        MC = 1,
        recordLabels = true,
        selectedLabels = sl,
        _labelsChoosen_pp,
        addEntropy = false,
    )

    @test !isPartial(P12)
    @test P12._partial === nothing
    # @test isapprox( mean(P12)[1], 0, atol=1 )
    # @test isapprox( mean(P12)[2], 0, atol=1 )

    @show sl;
    _labelsChoosen_pp

    # ensure all posterior product labels are from leaf nodes only
    sl1 = [s[1] for s in sl]
    sl2 = [s[2] for s in sl]

    @test all(l -> ApproxManifoldProducts.isLeaf_BTLabel(P1.belief, l), sl1)
    @test all(l -> ApproxManifoldProducts.isLeaf_BTLabel(P2.belief, l), sl2)

    # # check the sorting of the labels is consistent by rebuilding a shuffled belief
    # P1_ = manikde!(M, shuffle(pts1); bw = [1; 1.0])

    # @test all(s->s[1] ≈ s[2], zip(getPoints(P1), getPoints(P1_)) )

    P12

## validate selected labels are working properly, with addEntropy=false

    invpermute(B::ManellicTree, s::Int) = findfirst(==(s), B.permute)
    # use idx 1 assuming all leaf bandwidths are the same
    bw1 = getBW(P1)[invpermute(P1.belief,1)] .^ 2
    bw2 = getBW(P2)[invpermute(P2.belief,1)] .^ 2


    uhm = ApproxManifoldProducts.calcProductKernelsBTLabels(
        M,
        [P1.belief; P2.belief],
        [(sl1[1],sl2[1]);],
        false;
    )
    u1 = pts1[sl1[1] % N]
    u2 = pts2[sl2[1] % N]
    u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
    ???? calcProductGaussians is being fed different bandwidths, at least a square vs sqrt issue -- wip dedicated test for `[components...]` vs `[u...],[c...]`
    @test isapprox(mean(uhm[1]), u12)

    pts12 = getPoints(P12; permute=false)
    dropdups = Dict{Vector{Int},Int}()
    for sidx = 1:N
        # @info "debug" sidx sl1[sidx] sl2[sidx] 
        u1 = pts1[sl1[sidx] % N]
        u2 = pts2[sl2[sidx] % N]

        u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        
        # workaround for duplicate selections in pts12 test
        dropdups[sl[sidx]] = get(dropdups, sl[sidx], 0) + 1
        idxoff = 0
        for (k,i) in dropdups
            idxoff += i
        end
        # TODO test that kernel weights increase for each duplicate selection
        # @isapprox( getWeights(P12)[invpermute(P12.belief, sidx)], 1 / N * dropdups[sl[sidx]])
        @info "db" sidx idxoff pts12[idxoff] u12


        @test isapprox(u12, pts12[idxoff])
    end

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

    P1 = manikde!(TranslationGroup(3), pts1)
    P2 = manikde!(TranslationGroup(3), pts2)

    # P12 = P1 * P2
    P12 = manifoldProduct([P1; P2], LieGroups.TranslationGroup(3); legacy = false)

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


##