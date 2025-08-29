# test on partial products with SpecialEuclideanGroup(2; variant = :right)

using LieGroups
using LieGroups: TranslationGroup, submanifold_component
using ApproxManifoldProducts
using Test
using BSON
using Random

##

@testset "partial product with a SpecialEuclideanGroup(2; variant = :right)" begin
    ##

    datafile = joinpath(@__DIR__, "testdata", "partialtest.bson")
    # BSON.save(datafile, dict)
    data = BSON.load(datafile)
    # pts1, pts2 ∈ SE(2)
    pts1 = data[:dict][:pts1]
    pts2 = data[:dict][:pts2]

    randU = Float64[]
    randN = Float64[]

    len = length(pts1)

    # define test manifold
    M = SpecialEuclideanGroup(2; variant = :right)
    # M = TranslationGroup(2) × SpecialOrthogonalGroup(2)
    e0 = ArrayPartition([0.0; 0.0], [1 0; 0 1.0]) # identity_element(M)

    # p1_SE2_kde full SpecialEuclideanGroup(2; variant = :right)
    p1_SE2_kde = manikde!(M, pts1)
    p1_SE2_marg_Tr2_kde = marginal(p1_SE2_kde, [1; 2])

    # p2_SE2_marg_Tr2_kde only Translation(2) part
    p2_SE2_kde = manikde!(M, pts2)
    p2_SE2_marg_Tr2_kde = marginal(p2_SE2_kde, [1; 2])

    # product of full and marginal
    # p12 = p1_SE2_kde*p2_SE2_marg_Tr2_kde

    ## =======================FIRST PRODUCT================================================

    # this function also exports which kernels (ie labels) from incoming densities should be multiplied
    selectedLabels = Vector{Vector{Int}}()
    # multiply full SE2 p1_SE2_kde with translation-only marginal p2_SE2_marg_Tr2_kde 
    Random.seed!(0)
    p12 = manifoldProduct(
        [p1_SE2_kde; p2_SE2_marg_Tr2_kde];
        addEntropy = false,
        recordLabels = true,
        selectedLabels = selectedLabels,
        _randU = randU,
        _randN = randN,
    )
    #
    selectedLabels
    #

    @test !isPartial(p12)

    p12_ = marginal(p12, [1; 2])
    @test isPartial(p12_)

    _p12_ = manikde!(TranslationGroup(2), getPoints(p12_))

    ## intermediate test, check product of selected kernels match what is in the marginal

    for sidx = 1:len
        bw1 = getBW(p1_SE2_kde)[:, 1] .^ 2
        bw2 = getBW(p2_SE2_marg_Tr2_kde, false)[:, 1] .^ 2

        u1 = pts1[selectedLabels[sidx][1]]
        u2 = pts2[selectedLabels[sidx][2]]

        #FIXME - JT - I think this is comparing independent coordinates to a coupled SE(2) product
        # Perhaps fix and test: TranslationGroup(2) × SpecialOrthogonalGroup(2)
        # u12 = calcProductGaussians(TranslationGroup(2) × SpecialOrthogonalGroup(2), [u1,u2], [bw1,bw2])
        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])

        @test_broken isapprox(
            submanifold_component(mean(u12), 1),
            submanifold_component(getPoints(p12)[sidx], 1),
        )
    end

    ## now check the marginal dimensions only

    # now do submanifold dimensions separately as reference test -- should get a similar result
    pts1_ = getPoints(p1_SE2_marg_Tr2_kde)
    pts2_ = getPoints(p2_SE2_marg_Tr2_kde)

    ## Do the translation part separate
    for sidx = 1:len
        bw1 = getBW(p1_SE2_marg_Tr2_kde)[:, 1] .^ 2
        bw2 = getBW(p2_SE2_marg_Tr2_kde)[:, 1] .^ 2

        u1 = pts1_[selectedLabels[sidx][1]]
        u2 = pts2_[selectedLabels[sidx][2]]

        u12 = calcProductGaussians(TranslationGroup(2), [u1, u2], [bw1, bw2])
        @test isapprox(mean(u12), submanifold_component(M, getPoints(p12)[sidx], 1))
    end

    ## =======================SECOND PRODUCT================================================
    ## NEW PRODUCT ON ONLY THE TRANSLATION PART, COMPARE WITH ABOVE RESULT

    p1__ = manikde!(TranslationGroup(2), pts1_)
    p2__ = manikde!(TranslationGroup(2), pts2_)

    # p12__ = p1__*p2__
    selectedLabels__ = Vector{Vector{Int}}()
    p12__ = manifoldProduct(
        [p1__; p2__];
        addEntropy = false,
        recordLabels = true,
        selectedLabels = selectedLabels__,
        _randU = randU,
        _randN = randN,
    )
    #
    selectedLabels__

    ## compare calcProduct of full*partial with selection from partial*partial

    sidx = 1
    for sidx = 1:len
        bw1 = getBW(p1_SE2_kde)[:, 1] .^ 2
        bw2 = getBW(p2_SE2_marg_Tr2_kde, false)[:, 1] .^ 2

        # full-partial points, but selected from partial-partial product
        u1 = pts1[selectedLabels__[sidx][1]]
        u2 = pts2[selectedLabels__[sidx][2]]

        # same as above
        #FIXME - JT - I think this is comparing independent coordinates to a coupled SE(2) product
        # Perhaps fix and test: TranslationGroup(2) × SpecialOrthogonalGroup(2)
        # u12 = calcProductGaussians(TranslationGroup(2) × SpecialOrthogonalGroup(2), [u1,u2], [bw1,bw2])
        u12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        u12_ = calcProductGaussians(
            TranslationGroup(2),
            [submanifold_component(u1, 1), submanifold_component(u2, 1)],
            [bw1[1:2], bw2[1:2]],
        )

        @test_broken isapprox(submanifold_component(mean(u12), 1), mean(u12_); atol = 0.001) # atol = 0.1
        @test isapprox(getPoints(p12__)[sidx], mean(u12_))
    end

    ##

    @test mmd(_p12_, p12__) < 0.001

    ##
end

## DEBUG PLOTTING ==========================================================================

## Plots showing the problem, p12 was wrong!!

# using Cairo, RoMEPlotting
# Gadfly.set_default_plot_size(35cm,20cm)

# n=10; plotKDE([p1_SE2_marg_Tr2_kde;p2_SE2_marg_Tr2_kde; p12], levels=3, selectedPoints=selectedLabels[n:n])
# n=3; plotKDE([p1__;p2__; p12__], levels=3, selectedPoints=selectedLabels__[n:n])
# plotKDE([p1__; p2__; p12__])

# plotKDE([_p12_; p12__])

##
