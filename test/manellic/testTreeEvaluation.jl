
# using Revise
using Test
using ApproxManifoldProducts
using Random
using LinearAlgebra
using StaticArrays
using TensorCast
# using Manifolds
using LieGroups
import Rotations as Rot_
using Distributions
import ApproxManifoldProducts: ManellicTree, eigenCoords, splitPointsEigen

using Optim

using JSON3

DATADIR = joinpath(dirname(@__DIR__), "testdata")

##


@error "FIXME: add test building Manellic tree on edge case where all points are 0 with defined bandwidth"


@testset "Manellic basic evaluation test 1D" begin
    ##

    M = TranslationGroup(1)
    pts = [[randn();] for _ = 1:100]
    bw = ones(1, 1)
    kType = AMP.MvNormalKernel
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = kType,
    )

    @test mtree.tree_kernels[1] isa kType
    @test mtree.tree_kernels[2] isa kType
    @test mtree.tree_kernels[3] isa kType

    @test isapprox(pdf(Normal(0, 1), 0), AMP.evaluate(mtree, SA[0.0;]); atol=0.15)

    @error "expectedLogL for different number of test points not working yet."
    # AMP.expectedLogL(mtree, [randn(1) for _ in 1:5])

    @test 0.5 < AMP.entropy(mtree)

    # Vector bw required for backward compat with legacy belief structure
    mtreeV = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = [1.0;],
        kernel = AMP.MvNormalKernel,
    )

    bel = manikde!(
        M,
        pts;
        bw,
        belmodel = (a, b, aF, dF) -> ApproxManifoldProducts.buildTree_Manellic!(
            M,
            pts;
            kernel_bw = b,
            kernel = AMP.MvNormalKernel,
        ),
    )

    @test isapprox(0.4, bel([0.0;]); atol = 0.1)

    ##
end

@testset "Manellic tree bandwidth evaluation" begin
    ## load know test data test

    json_string = read(joinpath(DATADIR, "manellic_test_data.json"), String)
    dict = JSON3.read(json_string, Dict{Symbol, Vector{Float64}})

    M = TranslationGroup(1)
    pts = [[v;] for v in dict[:evaltest_1_pts]]
    bw = reshape(dict[:evaltest_1_bw], 1, 1)
    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = AMP.MvNormalKernel,
    )

    AMP.expectedLogL(mtree, pts)

    @test AMP.expectedLogL(mtree, pts) < Inf

    # to enable faster bandwidth selection/optimization
    ekr = ApproxManifoldProducts.getKernelLeaf(mtree, 1, false)
    ekr_ = ApproxManifoldProducts.updateKernelBW(ekr, SA[1.0;;])

    @test typeof(ekr) == typeof(ekr_)

    # confirm that updating the bandwidths works properly
    Σ = [0.1 + 0.5 * rand();;]

    mtr = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = Σ,
        kernel = AMP.MvNormalKernel,
    )
    mtr_ = ApproxManifoldProducts.updateBandwidths(mtree, Σ)

    # 
    @test isapprox(mtr([0.0]), mtr_([0.0]); atol = 1e-10)
    @test isapprox(
        ApproxManifoldProducts.entropy(mtr),
        ApproxManifoldProducts.entropy(mtr_);
        atol = 1e-10,
    )

    ##
end


@testset "Manellic tree evaluation check for NaN" begin
##

M = LieGroups.TranslationGroup(2)
u = [0.21651994984010028, 100.1499400950606]
c = [1.225550648840573 0.24068320456978248; 0.24068320456978248 1.005129493645892]

tmp_product = ApproxManifoldProducts.MvNormalKernel(u, c)
eval_at_points = [[-0.0590062185192033, -0.15598788922416723],]

smw = ApproxManifoldProducts.evaluateDensityAtPoints(M, tmp_product, eval_at_points, true) # TBD: smw = evaluate(tmp_product, )
# eval with normalize=true forces sum of evals to be 1, as this is often used for categorical sampling
@test length(smw) == 1
@test !isnan(smw[1])

##
end