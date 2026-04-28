
using Test
using ApproxManifoldProducts
using LieGroups
using Manifolds
using LinearAlgebra
using Random


##

@testset "test getManifoldPartial on Euclidean(N)" begin

##

    M = LieGroups.TranslationGroup(3)

    @test getManifoldPartial(M, [1; 2; 3])[1] == LieGroups.TranslationGroup(3)
    @test getManifoldPartial(M, [2; 3])[1] == LieGroups.TranslationGroup(2)

    @test getManifoldPartial(M, [1; 2; 3], zeros(3))[1] == LieGroups.TranslationGroup(3)
    @test isapprox(getManifoldPartial(M, [1; 2; 3], zeros(3))[2], zeros(3))

    @test getManifoldPartial(M, [2; 3], zeros(3))[1] == LieGroups.TranslationGroup(2)
    @test isapprox(getManifoldPartial(M, [2; 3], zeros(3))[2], zeros(2))

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

@testset "test getManifoldPartial on LieGroups.ProductLieGroup" begin
##

    M = LieGroups.TranslationGroup(2) × SpecialOrthogonalGroup(2)

    @test getManifoldPartial(M, [1; 2; 3])[1] ==
          LieGroups.TranslationGroup(2) × SpecialOrthogonalGroup(2)

    @test getManifoldPartial(M, [1;])[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [2;])[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [1; 2])[1] == LieGroups.TranslationGroup(2)

    @test getManifoldPartial(M, [3;])[1] == SpecialOrthogonalGroup(2)

    @test getManifoldPartial(M, [1; 3])[1] ==
          LieGroups.ProductLieGroup(LieGroups.TranslationGroup(1), SpecialOrthogonalGroup(2))

    repr = ArrayPartition([0.0; 0], [1 0; 0 1.0])

    @test getManifoldPartial(M, [1; 2; 3], repr)[1] ==
          LieGroups.TranslationGroup(2) × SpecialOrthogonalGroup(2)
    @test getManifoldPartial(M, [1; 2; 3], repr)[2] == repr

    @test getManifoldPartial(M, [1;], repr)[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [1;], repr)[2] == [0.0;]

    @test getManifoldPartial(M, [2;], repr)[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [2;], repr)[2] == [0.0;]

    @test getManifoldPartial(M, [1; 2], repr)[1] == LieGroups.TranslationGroup(2)
    @test getManifoldPartial(M, [1; 2], repr)[2] == [0.0; 0]

    @test getManifoldPartial(M, [3;], repr)[1] == SpecialOrthogonalGroup(2)
    @test getManifoldPartial(M, [3;], repr)[2].x[1] == submanifold_component(repr, 2)

    @test getManifoldPartial(M, [1; 3], repr)[1] ==
          LieGroups.ProductLieGroup(LieGroups.TranslationGroup(1), SpecialOrthogonalGroup(2))
    r_repr = getManifoldPartial(M, [1; 3], repr)[2]
    @test r_repr isa ArrayPartition
    @test submanifold_component(r_repr, 1) == [0.0;]
    @test submanifold_component(r_repr, 2) == [1 0; 0 1.0]

##
end

@testset "test getManifoldPartial on SpecialEuclideanGroup(2; variant = :right)" begin
##

    M = SpecialEuclideanGroup(2; variant = :right)

    @test getManifoldPartial(M, [1; 2; 3])[1] == M

    @test getManifoldPartial(M, [1;])[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [2;])[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [1; 2])[1] == LieGroups.TranslationGroup(2)

    @test getManifoldPartial(M, [3;])[1] == SpecialOrthogonalGroup(2)

    @test getManifoldPartial(M, [1; 3])[1] ==
          LieGroups.ProductLieGroup(LieGroups.TranslationGroup(1), SpecialOrthogonalGroup(2))

    repr = ArrayPartition([0.0; 0], [1 0; 0 1.0])

    @test getManifoldPartial(M, [1; 2; 3], repr)[1] == M
    @test getManifoldPartial(M, [1; 2; 3], repr)[2] == repr

    @test getManifoldPartial(M, [1;], repr)[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [1;], repr)[2] == [0.0;]

    @test getManifoldPartial(M, [2;], repr)[1] == LieGroups.TranslationGroup(1)
    @test getManifoldPartial(M, [2;], repr)[2] == [0.0;]

    @test getManifoldPartial(M, [1; 2], repr)[1] == LieGroups.TranslationGroup(2)
    @test getManifoldPartial(M, [1; 2], repr)[2] == [0.0; 0]

    @test getManifoldPartial(M, [3;], repr)[1] == SpecialOrthogonalGroup(2)
    @test getManifoldPartial(M, [3;], repr)[2].x[1] == submanifold_component(repr, 2)

    @test getManifoldPartial(M, [1; 3], repr)[1] ==
          LieGroups.ProductLieGroup(LieGroups.TranslationGroup(1), SpecialOrthogonalGroup(2))
    r_repr = getManifoldPartial(M, [1; 3], repr)[2]
    @test r_repr isa ArrayPartition
    @test submanifold_component(r_repr, 1) == [0.0;]
    @test submanifold_component(r_repr, 2) == [1 0; 0 1.0]

##
end

@testset "Reminder, getManifoldPartial on Sphere(2) [TBD]" begin

##

    @error "Must fix Sphere(2) partial test"
    @test_broken false

##
end


@testset "test replace (not replace!) overloads full and partial/marginal" begin
##

    N = 10
    M = LieGroups.TranslationGroup(3)
    pts0 = [zeros(3) for _ = 1:N]
    # X0 = manikde!(M, pts0; bw = diagm(0.001*ones(3)))
    @error "RESTORE manikde test forced bandwidth for repeat points non-PosDefCovariance"
    @test_broken false

    pts = [randn(3) for _ = 1:N]
    X = manikde!(M, pts)

##

    # X_ = replace(X0, X)
    gpts = getPoints(X)
    @test N == length(gpts)
    # @test isapprox(X_, X)

##

    X = manikde!(M, pts; partial = [1; 3])
    @error "restore tests for manikde partials"
    # X_ = replace(X0, X)

##

    # # check metadata
    # @test isapprox(getBW(X_, false)[[1; 3], 1], getBW(X, false)[[1; 3], 1])
    # @test !isapprox(getBW(X_, false)[[1; 3], 1], getBW(X0, false)[[1; 3], 1])

    # @test isapprox(X_.infoPerCoord[[1; 3]], X.infoPerCoord[[1; 3]])

    # @test !isPartial(X_)

    # # check points
    # x0 = getPoints(X0)
    # x = getPoints(X, false)
    # x_ = getPoints(X_)
    # for (i, pt) in enumerate(x0)
    #     # partial of X does not replace 
    #     @test isapprox(pt[2], x_[i][2])
    #     @test isapprox(x[i][[1; 3]], x_[i][[1; 3]])
    # end

    # ## must also test replace for partial into different partial

    # pts3 = [randn(3) for _ = 1:N]
    # X3 = manikde!(M, pts3; partial = [3;])
    # # and replace partial/marginal values
    # X__ = replace(X, X3)

    # @test isPartial(X__)
    # @test getPartial(X__) == [1; 3]

    # x__ = getPoints(X__, false)
    # for (i, pt) in enumerate(x)
    #     @test isapprox(pt[1:2], x__[i][1:2])
    #     @test isapprox(pts3[i][3], x__[i][3])
    # end

    # ## union of two partials over all dimensions should drop the partial status

    # pts12 = [randn(3) for _ = 1:N]
    # X12 = manikde!(M, pts12; partial = [1; 2], infoPerCoord = 2 * ones(3))

    # X_np = replace(X12, X3)

    # @test !isPartial(X_np)
    # @test isapprox(X_np.infoPerCoord, [2; 2; 1])

##
end

#


@testset "test getPoints of marginal with representation on LieGroups.ProductLieGroup" begin

##

    N = 100
    # M = SpecialEuclideanGroup(2; variant = :right)
    M = LieGroups.TranslationGroup(2) × SpecialOrthogonalGroup(2)
    u0 = ArrayPartition([0.0; 0], [1 0; 0 1.0])

    pts = [exp(M, u0, hat(LieAlgebra(M), [10 .+ randn(2); randn()])) for i = 1:N]

##

    P = manikde!(M, pts)

##

    P12 = marginal(P, [1; 2])

    p12 = getPoints(P12)

    @test length(p12) == N
    @test length(p12[1]) == 2
    @test getManifold(P12, true) == LieGroups.TranslationGroup(2)

##
end

@testset "test getPoints of marginal with representation on SE2" begin

##

    N = 100
    M = SpecialEuclideanGroup(2; variant = :right)
    u0 = ArrayPartition([0.0; 0], [1 0; 0 1.0])

    pts = [exp(M, hat(LieAlgebra(M), [10 .+ randn(2); randn()])) for i = 1:N]

    P = manikde!(M, pts)

    P12 = marginal(P, [1; 2])

    p12 = getPoints(P12)

    @test length(p12) == N
    @test length(p12[1]) == 2
    @test getManifold(P12, true) == LieGroups.TranslationGroup(2)

##
end



@testset "Tree reconstruction of 1D data as a either 2->[x *] or [* y] partials" begin
##

    # test lifted from (non-partial) tree construction test file
    M = LieGroups.TranslationGroup(2)
    # already sorted list
    pts = [[1.0; NaN], [2.0; NaN], [4.0; NaN], [7.0; NaN], [11.0; NaN], [16.0; NaN], [22.0; NaN]]
    bw = [1.0; 0.0]
    N = length(pts)
    partial = (1,)
    M_, reprl, partl_cb = ApproxManifoldProducts.getManifoldPartial(M, partial, pts[1])

##

    # preemptively check splitPoints 
    begin
        
        ax_CCp, mask, _p, _bw = ApproxManifoldProducts.splitPointsEigen(
            M,
            pts;
            kernel_bw = bw,
            partial,
        )

        @test mask[1:4] == BitVector([0,0,0,0])
        @test mask[5:7] == BitVector([1,1,1])
    end

##

    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
        partial,
    )

##

    @test mtree.geometric_permute[1] == [1, 2, 3, 4, 5, 6, 7]
    @test 9.0 ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[1]
    @test isnan(mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[2])

## shuffle

    perm = shuffle(1:length(pts))
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts[perm];
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
        partial,
    )

    if mtree.geometric_permute[1] == perm
        @test true
    else
        @error "Unreliable permute test, FIXME for consistent results"
    end
    @test 9.0 ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[1]
    @test isnan(mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[2])

##

    # test lifted from (non-partial) tree construction test file
    M = LieGroups.TranslationGroup(2)
    # already sorted list
    pts = [[NaN; 1.0], [NaN; 2.0], [NaN; 4.0], [NaN; 7.0], [NaN; 11.0], [NaN; 16.0], [NaN; 22.0]]
    bw = [0.0; 1.0]
    N = length(pts)
    partial = (2,)
    M_, reprl, partl_cb = ApproxManifoldProducts.getManifoldPartial(M, partial, pts[1])

    # preemptively check splitPoints 
    begin
        
        ax_CCp, mask, _p, _bw = ApproxManifoldProducts.splitPointsEigen(
            M,
            pts;
            kernel_bw = bw,
            partial,
        )

        @test mask[1:4] == BitVector([0,0,0,0])
        @test mask[5:7] == BitVector([1,1,1])
    end

    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
        partial,
        partl_cb,
    )

##

    @test mtree.geometric_permute[1] == [1, 2, 3, 4, 5, 6, 7]
    @test isnan(mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[1])
    @test 9.0 ≈ mean(ApproxManifoldProducts.getKernelTree(mtree, 1))[2]


##
end

@testset "test marginal of marginal (partial) helper" begin
##

    M = LieGroups.TranslationGroup(3)
    pts = [randn(3) for _ = 1:75]

    X = manikde!(M, pts; partial = (1,3))

    X_ = marginal(X, [3])

    ps3 = getPoints(X_)

    for (i, pt) in enumerate(pts[X_.geometric_permute[1]])
        @test isapprox(ps3[i][1], pt[3])
    end

    try
        M = LieGroups.TranslationGroup(4)
        # check the constructor when only a few points are available
        X = manikde!(M, pts; partial = (1, 3, 4))
    catch
        @test_broken false
    end

##
end

##