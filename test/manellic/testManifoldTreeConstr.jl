
# using Revise
using Test
using ApproxManifoldProducts
using Random
import Statistics
using LinearAlgebra
using StaticArrays
using TensorCast
using Manifolds
using LieGroups
import Rotations as Rot_
using Distributions

using Optim

using JSON3

##

DATADIR = joinpath(dirname(@__DIR__), "testdata")

# test 
function testEigenCoords!(r_C = pi / 3, ax_CC = [SA[5 * randn(); randn()] for _ = 1:100])
    M = LieGroups.TranslationGroup(2)
    _R(α, s = exp(-α * im)) = real(s) * SA[1 0; 0 1] + imag(s) * SA[0 1; -1 0]
    # _R(α) = SA[cos(α) sin(α); -sin(α) cos(α)]
    r_R_ax = _R(r_C)
    # rotate coordinates
    r_CC = map(ax_CC) do ax_C
        r_R_ax * ax_C + SA[10; -100]
    end
    r_CV = Statistics.cov(M, r_CC)
    r_R_ax_, L, pidx = ApproxManifoldProducts.eigenCoords!(r_CV)

    # spot check
    @show _ax_ERR = log(SpecialOrthogonalGroup(2), (r_R_ax_') * r_R_ax)[1, 2]
    @show testval = isapprox(0, _ax_ERR; atol = 8 / length(ax_CC))
    @assert testval "Spot check failed on eigen split of manifold points, the estimated point rotation matrix did not match construction. length(ax_CC)=$(length(ax_CC))"

    return r_CC, r_R_ax_, pidx, r_CV
end


function testMDEConstr(
    pts::AbstractVector{<:AbstractVector{<:Real}},
    permref = sortperm(pts; by = s -> getindex(s, 1));
    lseg = 1:2,
    rseg = 3:4,
    atol = 1e-6,
)
    # check permutation
    M = LieGroups.TranslationGroup(1)
    bw = [1.0]

    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )
    @test permref == mtree.permute
    @test isapprox(mean(M, pts), mean(mtree.tree_kernels[1]); atol = 1e-10)
    @test Set(mtree.segments[1]) == Set(union(lseg, rseg))
    @test Set(mtree.segments[2]) == Set(mtree.permute[lseg])
    @test Set(mtree.segments[3]) == Set(mtree.permute[rseg])
    @test isapprox(mean(M, pts[mtree.permute[lseg]]), mean(mtree.tree_kernels[2]); atol)
    @test isapprox(mean(M, pts[mtree.permute[rseg]]), mean(mtree.tree_kernels[3]); atol)
    return nothing
end



## ===================================================================


@testset "HomotopyDensity 1D unbalanced tree construction, left and sorted" begin
## 

    M = LieGroups.TranslationGroup(1)
    # design mean at 0.0
    pts = [
        [-1.0],
        [3.0],
        [-2.0],
    ]
    
    bw = [0.5;]
    #
    #
    #               {1}1:3
    #              /      \
    #         {2}1,3      (3)2
    #          /   \      /   \
    #       (4)3  (5)1   *     *
    #
    hode = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

##

    @test 1 == Ndim(hode)
    @test 3 == Npts(hode)

    @test hode.permute == [3;1;2]
    @test hode.segments[1] == Set(1:3)
    @test hode.segments[2] == Set([1,3]) # segments are raw dataidx, not permuted dataidx
    @test !isassigned(hode.segments, 3) # no third segment because right child is leaf

    @test isassigned(hode.tree_kernels, 1)
    @test isassigned(hode.tree_kernels, 2)
    @test !isassigned(hode.tree_kernels, 3)

    @test isapprox( 0.0, mean(hode.tree_kernels[1])[1]; atol = 1e-6)
    @test isapprox(-1.5, mean(hode.tree_kernels[2])[1]; atol = 1e-6)

    @test isassigned(hode.leaf_kernels, 1)
    @test isassigned(hode.leaf_kernels, 2)
    @test isassigned(hode.leaf_kernels, 3)

    # leaf kernels are sorted in geometric order along eigen axis
    @test isapprox(-2.0, mean(hode.leaf_kernels[1])[1]; atol = 1e-6)
    @test isapprox(-1.0, mean(hode.leaf_kernels[2])[1]; atol = 1e-6)
    @test isapprox( 3.0, mean(hode.leaf_kernels[3])[1]; atol = 1e-6)

    @test !ApproxManifoldProducts.isLeaf_BTLabel(hode, 1)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(hode, 2)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 3)

    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 4)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 5)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 6) # there for binary tree defaults, although undef
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 7) # there for binary tree defaults, although undef

    @test ApproxManifoldProducts.exists_BTLabel(hode, 1)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 2)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 3)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 4)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 5)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 6)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 7)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 8) # why is this here

##
end


@testset "HomotopyDensity 1D unbalanced tree construction, left but shuffled" begin
## 

    M = LieGroups.TranslationGroup(1)
    # design mean at 0.0
    pts = [
        [-2.0],
        [-1.0],
        [3.0],
    ]
    
    bw = [0.5;]
    #
    #
    #               {1}1:3
    #              /      \
    #         {2}1:2      (3)3
    #          /   \      /   \
    #       (4)1  (5)2   *     *
    #
    hode = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

##

    @test 1 == Ndim(hode)
    @test 3 == Npts(hode)

    @test hode.permute == [1;2;3]
    @test hode.segments[1] == Set(1:3)
    @test hode.segments[2] == Set(1:2)
    @test !isassigned(hode.segments, 3) # no third segment because right child is leaf

    @test isassigned(hode.tree_kernels, 1)
    @test isassigned(hode.tree_kernels, 2)
    @test !isassigned(hode.tree_kernels, 3)

    @test isapprox( 0.0, mean(hode.tree_kernels[1])[1]; atol = 1e-6)
    @test isapprox(-1.5, mean(hode.tree_kernels[2])[1]; atol = 1e-6)

    @test isassigned(hode.leaf_kernels, 1)
    @test isassigned(hode.leaf_kernels, 2)
    @test isassigned(hode.leaf_kernels, 3)

    @test isapprox(-2.0, mean(hode.leaf_kernels[1])[1]; atol = 1e-6)
    @test isapprox(-1.0, mean(hode.leaf_kernels[2])[1]; atol = 1e-6)
    @test isapprox( 3.0, mean(hode.leaf_kernels[3])[1]; atol = 1e-6)

    @test !ApproxManifoldProducts.isLeaf_BTLabel(hode, 1)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(hode, 2)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 3)

    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 4)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 5)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 6) # there for binary tree defaults, although undef
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 7) # there for binary tree defaults, although undef

    @test ApproxManifoldProducts.exists_BTLabel(hode, 1)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 2)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 3)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 4)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 5)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 6)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 7)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 8)

##
end


@testset "HomotopyDensity 1D unbalanced tree construction, right" begin
## 

    M = LieGroups.TranslationGroup(1)
    # design mean at 0.0
    pts = [
        [1.0],
        [2.0],
        [-3.0],
    ]
    
    bw = [0.5;]
    #
    #
    #               {1}1:3
    #              /      \
    #          (2)3      {3}1:2
    #          /   \     /     \
    #         *     *  (6)1    (7)2
    #
    hode = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

##

    @test 1 == Ndim(hode)
    @test 3 == Npts(hode)

    @test hode.permute == [3;1;2]
    @test hode.segments[1] == Set(1:3)
    @test !isassigned(hode.segments, 2) # no third segment because right child is leaf
    @test hode.segments[3] == Set(1:2)

    @test isassigned(hode.tree_kernels, 1)
    @test !isassigned(hode.tree_kernels, 2)
    @test isassigned(hode.tree_kernels, 3)

    @test isapprox( 0.0, mean(hode.tree_kernels[1])[1]; atol = 1e-6)
    @test isapprox( 1.5, mean(hode.tree_kernels[3])[1]; atol = 1e-6)

    @test isassigned(hode.leaf_kernels, 1)
    @test isassigned(hode.leaf_kernels, 2)
    @test isassigned(hode.leaf_kernels, 3)

    @test isapprox(-3.0, mean(hode.leaf_kernels[1])[1]; atol = 1e-6)
    @test isapprox( 1.0, mean(hode.leaf_kernels[2])[1]; atol = 1e-6)
    @test isapprox( 2.0, mean(hode.leaf_kernels[3])[1]; atol = 1e-6)

    @test !ApproxManifoldProducts.isLeaf_BTLabel(hode, 1)
    @test_broken ApproxManifoldProducts.isLeaf_BTLabel(hode, 2)
    @test_broken !ApproxManifoldProducts.isLeaf_BTLabel(hode, 3)

    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 4)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 5)
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 6) # there for binary tree defaults, although undef
    @test ApproxManifoldProducts.isLeaf_BTLabel(hode, 7) # there for binary tree defaults, although undef

    @test ApproxManifoldProducts.exists_BTLabel(hode, 1)
    @test !ApproxManifoldProducts.exists_BTLabel(hode, 2)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 3)
    @test ApproxManifoldProducts.exists_BTLabel(hode, 4) # not sure about this
    @test_broken !ApproxManifoldProducts.exists_BTLabel(hode, 5)
    # @test ApproxManifoldProducts.exists_BTLabel(hode, 6) # not sure about this
    @test_broken ApproxManifoldProducts.exists_BTLabel(hode, 7)
    @test_broken ApproxManifoldProducts.exists_BTLabel(hode, 8)

##
end


@testset "HomotopyDensity 1D basic sorting of points with shuffle" begin
## 

    M = LieGroups.TranslationGroup(1)
    # pts = [randn(1) for _ = 1:5]
    pts = [
        [0.07322299439163212],
        [2.065271709556179],
        [-0.21699662409315343],
        [0.3625358873858872],
        [-0.7988970559113724],
    ]

    refperm = sortperm(pts)
    
    bw = [0.1;]
    #
    #
    #               {1}1:5
    #              /      \
    #          (2)135     {3}24
    #          /   \     /     \
    #         *     *  (6)1    (7)2
    #
    hode = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

    shf = shuffle(1:length(pts))
    hode_ = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts[shf];
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

##

    @test 5 == Npts(hode)
    
    @test all(refperm .== hode.permute)
    @test all(refperm .== shf[hode_.permute])
    @test all(hode.permute .== shf[hode_.permute])

    @test hode.segments[1] == Set(1:5)
    @test hode.segments[2] == Set([1,3,5])
    @test hode.segments[3] == Set([2,4])
    


##


    @error "expand sorting test to trivial TranslateGroup(2) with pts = [[*; 0], ...] producing same mtree.permute"
    @error "expand sorting test to trivial TranslateGroup(2) with pts = [[0; *], ...] producing same mtree.permute"

##
end


@testset "HomotopyDensity construction 1D" begin
##

    M = LieGroups.TranslationGroup(1)
    # already sorted list
    pts = [[1.0], [2.0], [4.0], [7.0], [11.0], [16.0], [22.0]]
    bw = [1.0]
    N = length(pts)
    
    # preemptively check splitPoints 
    begin
        
        @test isapprox([9.0;], Statistics.mean(pts))
        
        ax_CCp, mask, knl = ApproxManifoldProducts.splitPointsEigen(
            M,
            pts,
            1/7*ones(length(pts));
            kernel = ConcentratedGaussianKernel,
            kernel_bw = bw,
        )

        @test mask[1:4] == BitVector([0,0,0,0])
        @test mask[5:7] == BitVector([1,1,1])
    end

    #
    #               {1}1:7
    #              /      \
    #        {2}1:4        {3}5:7
    #        /    \         /    \
    #    {4}1:2  {5}3:4  {6}5:6   (7)7
    #    /  \    /  \     /  \     /  \
    #   (8)(9) (10)(11) (12)(13)  *    *
    #    1  2    3  4     5  6
    #
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

##
    @test mtree.permute == [1;2;3;4;5;6;7]

    @test 7 == length(intersect(mtree.segments[1], Set(1:7))) # root is parent to all
    @test 4 == length(intersect(mtree.segments[2], Set(1:4))) # first left is parent to 1:4
    @test 3 == length(intersect(mtree.segments[3], Set(5:7))) # first right is parent to 5:7
    @test 2 == length(intersect(mtree.segments[4], Set(1:2))) # second left is parent to 1:2
    @test 2 == length(intersect(mtree.segments[5], Set(3:4))) # second right is parent to 3:4
    @test 2 == length(intersect(mtree.segments[6], Set(5:6))) # third left is parent to 5:6
    @test !isassigned(mtree.segments, 7)                      # third right is unused
    
    @test isapprox(mean(M, pts),      mean(mtree.tree_kernels[1]); atol = 1e-6)
    @test isapprox(mean(M, pts[1:4]), mean(mtree.tree_kernels[2]); atol = 1e-6)
    @test isapprox(mean(M, pts[5:7]), mean(mtree.tree_kernels[3]); atol = 1e-6)
    @test isapprox(mean(M, pts[1:2]), mean(mtree.tree_kernels[4]); atol = 1e-6)
    @test isapprox(mean(M, pts[3:4]), mean(mtree.tree_kernels[5]); atol = 1e-6)
    @test isapprox(mean(M, pts[5:6]), mean(mtree.tree_kernels[6]); atol = 1e-6)

    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 1)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 2)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 3)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 4)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 5)
    @test !ApproxManifoldProducts.isLeaf_BTLabel(mtree, 6)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 7)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 8)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 9)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 10)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 11)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 12)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 13)
    @test ApproxManifoldProducts.isLeaf_BTLabel(mtree, 14)

    # check leaf nodes
    @test [1.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 8))
    @test [2.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 9))
    @test [4.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 10))
    @test [7.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 11))
    @test [11.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 12))
    @test [16.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 13))
    @test [22.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 14))
    # 7 is 14
    @test [22.0;] ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 7))
    
##

    @test ApproxManifoldProducts.exists_BTLabel(mtree, floor(Int, N / 2))
    @test ApproxManifoldProducts.exists_BTLabel(
        mtree,
        ApproxManifoldProducts.leftIndex(mtree, floor(Int, N / 2)),
    )
    @test !ApproxManifoldProducts.exists_BTLabel(mtree, 2 * N + 1)


## test sorting of labels is consistent by rebuilding a shuffled belief
    mtree_ = manikde!(M, shuffle(pts); bw)

    @test all(s->s[1] ≈ s[2], zip(getPoints(mtree), getPoints(mtree_)) )

## for 4 values

    # manual orders
    testMDEConstr([[0.0;], [1.0], [3.0;], [6.0;]])
    testMDEConstr([[0.0;], [1.0], [6.0;], [3.0;]])
    testMDEConstr([[0.0;], [3.0], [1.0;], [6.0;]])
    testMDEConstr([[1.0;], [0.0], [3.0;], [6.0;]])
    testMDEConstr([[1.0;], [0.0], [6.0;], [3.0;]])
    testMDEConstr([[1.0;], [6.0], [0.0;], [3.0;]])
    testMDEConstr([[6.0;], [1.0], [3.0;], [0.0;]])

    testMDEConstr([
        [0.9497270480266986;],
        [-0.5973125859935883;],
        [-0.6031001429225558;],
        [-0.3971695179687664;],
    ])

    # randomized orders for 4 values
    pts = [[0.0;], [1.0], [3.0;], [6.0;]]
    for i = 1:10
        testMDEConstr(pts[shuffle(1:4)])
    end

## for 5 values

    testMDEConstr([[0.0;], [1.0], [3.0;], [6.0;], [10.0;]]; lseg = 1:3, rseg = 4:5)
    testMDEConstr([[0.0;], [1.0], [6.0;], [3.0;], [10.0;]]; lseg = 1:3, rseg = 4:5)

    # randomized orders for 5 values
    pts = [[0.0;], [1.0], [3.0;], [6.0;], [10.0;]]
    for i = 1:10
        testMDEConstr(pts[shuffle(1:length(pts))]; lseg = 1:3, rseg = 4:5)
    end

## for 7 values

    # randomized orders for 7 values
    pts = [[0.0;], [1.0], [3.0;], [6.0;], [10.0;], [15.0;], [21.0;]]
    for i = 1:10
        testMDEConstr(pts[shuffle(1:length(pts))]; lseg = 1:4, rseg = 5:7)
    end

    #
    M = LieGroups.TranslationGroup(1)
    pts = [randn(1) for _ = 1:8]
    for i = 1:10
        _pts = pts[shuffle(1:length(pts))]
        testMDEConstr(_pts; lseg = 1:4, rseg = 5:8)
    end

##
end

@testset "test Manellic tree utilities w skeleton object" begin
##
    M = LieGroups.TranslationGroup(1)
    N = 32
    pts = [randn(1) for _ = 1:N]
    # weights = ones(N) ./ N
    KT = ConcentratedGaussianKernel
    KL = ConcentratedGaussianKernel
    lkern = Vector{KL}(undef, N)

##

    mtree = ApproxManifoldProducts.HomotopyDensity{
        nothing
    }(;
        manifold = M,
        data = pts,
        leaf_kernels = lkern,                           # leaf_kernels
        tree_kernels = Vector{KT}(undef, N),       # tree_kernels
    );
    
##
    # tree kernel indices
    @test 2 == ApproxManifoldProducts.leftIndex(mtree, 1)
    @test 3 == ApproxManifoldProducts.rightIndex(mtree, 1)
    
    @test 4 == ApproxManifoldProducts.leftIndex(mtree, 2)
    @test 5 == ApproxManifoldProducts.rightIndex(mtree, 2)

    @test 6 == ApproxManifoldProducts.leftIndex(mtree, 3)
    @test 7 == ApproxManifoldProducts.rightIndex(mtree, 3)

    @test 8 == ApproxManifoldProducts.leftIndex(mtree, 4)
    @test 9 == ApproxManifoldProducts.rightIndex(mtree, 4)

    @test 10 == ApproxManifoldProducts.leftIndex(mtree, 5)
    @test 11 == ApproxManifoldProducts.rightIndex(mtree, 5)

    @test 16 == ApproxManifoldProducts.leftIndex(mtree, 8)
    @test 17 == ApproxManifoldProducts.rightIndex(mtree, 8)

    # children are now leaf nodes (assuming first N=[1..32] are tree kernels, while [33..64] are leaf kernels)
    @test 33 == ApproxManifoldProducts.leftIndex(mtree, 16)
    @test 34 == ApproxManifoldProducts.rightIndex(mtree, 16)

    @test 35  == ApproxManifoldProducts.leftIndex(mtree, 17)
    @test 36 == ApproxManifoldProducts.rightIndex(mtree, 17)

    @test 64 == ApproxManifoldProducts.rightIndex(mtree, 31)


    # @test 11 == ApproxManifoldProducts.leftIndex(mtree, 5)
    # @test 12 == ApproxManifoldProducts.rightIndex(mtree, 5)
    # @test 13 == ApproxManifoldProducts.leftIndex(mtree, 6)
    # @test 14 == ApproxManifoldProducts.rightIndex(mtree, 6)
    # # but note these are not assigned
    # @test 15 == ApproxManifoldProducts.leftIndex(mtree, 7)
    # @test 16 == ApproxManifoldProducts.rightIndex(mtree, 7)

    # leaf kernel indices
    @test N + 1 == ApproxManifoldProducts.leftIndex(mtree, floor(Int, N / 2))
    @test N + 2 == ApproxManifoldProducts.rightIndex(mtree, floor(Int, N / 2))
    
##

    # TBD NOTE, maybe index should be a tuple of (level, node) instead of a single integer, (s=idx*2; (s % N, (s % N) + 1 ))

end


##
@testset "test HomotopyDensity construction" begin
##

    M = LieGroups.TranslationGroup(2)
    α = pi / 3
    r_CC, R, pidx, r_CV = testEigenCoords!(α)
    ax_CCp, mask, knl = ApproxManifoldProducts.splitPointsEigen(M, r_CC)
    @test sum(mask) == (length(r_CC) ÷ 2)
    @test knl isa ConcentratedGaussianKernel
    Mr = SpecialOrthogonalGroup(2)
    @test isapprox(α, vee(LieAlgebra(Mr), log(Mr, R))[1]; atol = 0.1)

##

    # using GLMakie
    # fig = Figure()
    # ax = Axis(fig[1,1])
    # ptsl = ax_CCp[mask]
    # ptsr = ax_CCp[xor.(mask,true)]
    # plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
    # plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
    # ax = Axis(fig[2,1])
    # ptsl = r_CC[mask]
    # ptsr = r_CC[xor.(mask,true)]
    # plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
    # plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
    # fig

## ensure that view of view can update original memory

    A = randn(3)
    A_ = view(A, 1:2)
    A__ = view(A_, 1:1)
    A__[1] = -100
    @test isapprox(-100, A[1]; atol = 1e-10)

##

    r_PP = r_CC # shortcut because we are in Euclidean space
    mtree = ApproxManifoldProducts.buildTree_Manellic!(M, r_PP; kernel = ConcentratedGaussianKernel)

    # test input data vs leaf kernels
    for i in eachindex(r_PP)
        @test isapprox(r_PP[i], getPoints(mtree, permute = false)[i])
        @test isapprox(r_PP[mtree.permute[i]], getPoints(mtree, permute = true)[i])
        # FIXME, test is useful but underneath is a yucky duplication of permuted raw data in leaf_kernels[]
        @test isapprox(r_PP[i], mean(ApproxManifoldProducts.getKernelLeaf(mtree, i, false)))
        @test isapprox(r_PP[mtree.permute[i]], mean(ApproxManifoldProducts.getKernelLeaf(mtree, i, true)))
    end

    @test all(isapprox.(mtree.weights[mtree.permute], getWeights(mtree, permute = true)))

##

    @cast pts[i, d] := r_PP[i][d]

    ptsl = pts[mtree.permute[1:50], :]
    ptsr = pts[mtree.permute[51:100], :]

##

    # fig = Figure()
    # ax = Axis(fig[1,1])

    # plot!(ax, ptsl[:,1], ptsl[:,2], color=:blue)
    # plot!(ax, ptsr[:,1], ptsr[:,2], color=:red)

    # fig

##

    ApproxManifoldProducts.evaluate(mtree, SA[10.0; -101.0])

##
end

@testset "HomotopyDensity 1D basic construction and evaluations" begin
## 

    M = LieGroups.TranslationGroup(1)
    pts = [randn(1) for _ = 1:128]
    mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel = ConcentratedGaussianKernel)

    ApproxManifoldProducts.evaluate(mtree, SA[0.0;])

## load know test data test

    json_string = read(joinpath(DATADIR, "manellic_test_data.json"), String)
    dict = JSON3.read(json_string, Dict{Symbol, Vector{Float64}})

##

    M = LieGroups.TranslationGroup(1)
    pts = [[v;] for v in dict[:evaltest_1_pts]]
    bw = reshape(dict[:evaltest_1_bw], 1, 1)
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )

    mtree.permute
    shf = shuffle(1:length(pts))
    mtree_ = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts[shf];
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
    )


    np = Normal(0, 1)
    h = 0.1
    xx = -5:h:5
    yy_ = pdf.(np, xx) # ref
    yy = [ApproxManifoldProducts.evaluate(mtree, [v;]) for v in xx] # test
    for (i, v) in enumerate(yy_)
        @test isapprox(v, yy[i]; atol = 0.05)
    end
    @test isapprox(1, sum(yy_ .* h); atol = 1e-3)
    @test isapprox(1, sum(yy .* h); atol = 1e-3)
    # using GLMakie
    # lines(xx, yy_, color=:red) # ref
    # lines!(xx, yy, color=:blue) # test

    # test sorting order of data 
    refperm = sortperm(pts)
    permref = sortperm(pts; by = s -> getindex(s, 1))

    @test all(refperm .== mtree.permute)
    @test all(refperm .== shf[mtree_.permute])
    @test all(mtree.permute .== shf[mtree_.permute])

    @test 0 == sum(permref - mtree.permute)

    @test 0 == sum(
        collect(sortperm(mtree.leaf_kernels; by = s -> mean(s))) -
        collect(1:length(mtree.leaf_kernels)),
    )

    #and leaf kernel sorting
    @test norm((pts[mtree.permute] .- mean.(mtree.leaf_kernels)) .|> s -> s[1]) < 1e-6

    # for (i,v) in enumerate(dict[:evaltest_1_at])
    #   # @show ApproxManifoldProducts.evaluate(mtree, [v;]), dict[:evaltest_1_dens][i]
    #   @test isapprox(dict[:evaltest_1_dens][i], ApproxManifoldProducts.evaluate(mtree, [v;]))
    # end
    # isapprox(dict[:evaltest_1_dens][5], ApproxManifoldProducts.evaluate(mtree, [dict[:evaltest_1_at][5]]))
    # eval test ref Normal(0,1)


    # check the sorting of the labels is consistent by rebuilding a shuffled belief
    mtree_ = manikde!(M, shuffle(pts); bw)

    @test all(s->s[1] ≈ s[2], zip(getPoints(mtree), getPoints(mtree_)) )

##
end

@testset "Test evaluate ConcentratedGaussianKernel" begin
##

    M = LieGroups.TranslationGroup(1)
    ker = ConcentratedGaussianKernel([0.0], [0.5;;])
    @test isapprox(ApproxManifoldProducts.evaluate(M, ker, [0.1]), pdf(MvNormal(mean(ker), cov(ker)), [0.1]))

    # Test wrapped cicular distribution 
    function pdf_wrapped_normal(μ, σ, θ; nwrap = 1000)
        s = 0.0
        for k = (-nwrap):nwrap
            s += exp(-(θ - μ + 2pi * k)^2 / (2 * σ^2))
        end
        return 1 / (σ * sqrt(2pi)) * s
    end

    M = LieGroups.CircleGroup(ℝ)
    ker = ConcentratedGaussianKernel([0.0], [0.1;;])
    @test isapprox(
        ApproxManifoldProducts.evaluate(M, ker, [0.1]),
        pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], 0.1),
    )

    ker = ConcentratedGaussianKernel([0], [2.0;;])
    @test isapprox(ApproxManifoldProducts.evaluate(M, ker, [0.0]), ApproxManifoldProducts.evaluate(M, ker, [2pi]))
    #TODO wrapped normal distributions broken
    @test_broken isapprox(
        pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], pi),
        ApproxManifoldProducts.evaluate(M, ker, [pi]),
    )
    @test_broken isapprox(
        pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], 0),
        ApproxManifoldProducts.evaluate(M, ker, [0.0]),
    )

##
    M = SpecialEuclideanGroup(2; variant = :right)
    ε = identity_element(M)
    Xc = [10, 20, 0.1]
    p = exp(M, hat(LieAlgebra(M), Xc))
    kercov = diagm([0.5, 2.0, 0.1] .^ 2)
    ker = ConcentratedGaussianKernel(p, kercov)
    @test isapprox(ApproxManifoldProducts.evaluate(M, ker, p), pdf(MvNormal(Xc, cov(ker)), Xc))

    Xc = [10, 22, -0.1]
    q = exp(M, hat(LieAlgebra(M), Xc))

    @test isapprox(pdf(MvNormal(cov(ker)), [0, 0, 0]), ApproxManifoldProducts.evaluate(M, ker, p))

    X = log(M, compose(M, inv(M, p), q))
    Xc_e = vee(LieAlgebra(M), X)
    pdf_local_coords = pdf(MvNormal(cov(ker)), Xc_e)

    @test isapprox(pdf_local_coords, ApproxManifoldProducts.evaluate(M, ker, q))

    delta_c = ApproxManifoldProducts.distanceMalahanobisCoordinates(M, ker, q)
    X = log(M, compose(M, inv(M, p), q))
    Xc_e = vee(LieAlgebra(M), X)
    malad_t = Xc_e' * inv(kercov) * Xc_e
    # delta_t = [10, 20, 0.1] - [10, 22, -0.1] 
    @test isapprox(malad_t, delta_c' * delta_c; atol = 1e-10)

    malad2 = ApproxManifoldProducts.distanceMalahanobisSq(M, ker, q)
    @test isapprox(malad_t, malad2; atol = 1e-10)

    rbfd = ApproxManifoldProducts.ker(M, ker, q, 0.5, ApproxManifoldProducts.distanceMalahanobisSq)
    @test isapprox(exp(-0.5 * malad_t), rbfd; atol = 1e-10)

    # NOTE 'global' distribution would have been 
    X = log(M, mean(ker), q)
    Xc_e = vee(LieAlgebra(M), X)
    pdf_global_coords = pdf(MvNormal(cov(ker)), Xc_e)

##
end

@testset "Basic HomotopyDensity manifolds construction and evaluations" begin
## 

    M = LieGroups.TranslationGroup(1)
    ε = identity_element(M)
    dis = MvNormal([3.0], diagm([1.0] .^ 2))
    Cpts = [rand(dis) for _ = 1:128]
    pts = map(c -> exp(M, ε, hat(LieAlgebra(M), c)), Cpts)
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = [0.2;;],
        kernel = ConcentratedGaussianKernel,
    )

##
    p = exp(M, ε, hat(LieAlgebra(M), [3.0]))
    y_amp = ApproxManifoldProducts.evaluate(mtree, p)

    y_pdf = pdf(dis, [3.0])

    @test isapprox(y_amp, y_pdf; atol = 0.1)

    # ps = [[p] for p = -0:0.01:6]
    # ys_amp = map(p->ApproxManifoldProducts.evaluate(mtree, exp(M, ε, hat(M, ε, p))), ps)
    # ys_pdf = pdf(dis, ps)

    # lines(first.(ps), ys_pdf)
    # lines!(first.(ps), ys_amp)

    # lines!(first.(ps), ys_pdf)
    # lines(first.(ps), ys_amp)
##

    M = SpecialOrthogonalGroup(2)
    ε = identity_element(M)
    dis = MvNormal([0.0], diagm([0.1] .^ 2))
    Cpts = [rand(dis) for _ = 1:128]
    pts = map(c -> exp(M, hat(LieAlgebra(M), c)), Cpts)
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = [0.005;;],
        kernel = ConcentratedGaussianKernel,
    )

##
    p = exp(M, ε, hat(LieAlgebra(M), [0.1]))
    y_amp = ApproxManifoldProducts.evaluate(mtree, p)

    y_pdf = pdf(dis, [0.1])

    @test isapprox(y_amp, y_pdf; atol = 0.55)

    ps = [[p] for p = -0.3:0.01:0.3]
    ys_amp = map(p -> ApproxManifoldProducts.evaluate(mtree, exp(M, ε, hat(LieAlgebra(M), p))), ps)
    ys_pdf = pdf(dis, ps)

    # lines(first.(ps), ys_pdf)
    # lines!(first.(ps), ys_amp)

    M = SpecialEuclideanGroup(2; variant = :right)
    ε = identity_element(M)
    dis = MvNormal([10, 20, 0.1], diagm([0.5, 2.0, 0.1] .^ 2))
    Cpts = [rand(dis) for _ = 1:128]
    pts = map(c -> exp(M, ε, hat(LieAlgebra(M), c)), Cpts)
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = diagm([0.05, 0.2, 0.01]),
        kernel = ConcentratedGaussianKernel,
    )

##
    p = exp(M, hat(LieAlgebra(M), [10, 20, 0.1]))
    y_amp = ApproxManifoldProducts.evaluate(mtree, p)
    y_pdf = pdf(dis, [10, 20, 0.1])
    # check kde eval is within 20% of true value
    y_err = y_amp - y_pdf
    @show y_pdf
    if !isapprox(0, y_err; atol = 0.2 * y_pdf)
        @warn "soft test failure for approx function vs. true Normal density function evaluation"
        @test_broken false
    end

##
end
