# test on partial products with SpecialEuclidean(2)

using Manifolds
using ApproxManifoldProducts
using Test
# using Random
using FileIO

##

@testset "partial product with a SpecialEuclidean(2)" begin
##

datafile = joinpath(@__DIR__, "testdata", "partialtest.jld2")
dict = load(datafile)
pts1 = dict["pts1"]
pts2 = dict["pts2"]


## define test manifold
M = SpecialEuclidean(2)
e0 = identity_element(M)

# p1 full SpecialEuclidean(2)
p1 = manikde!(M, pts1)
p1_ = marginal(p1,[1;2])

# p2 only Translation(2) part
_p2 = manikde!(M, pts2)
p2 = marginal(_p2, [1;2])

# product of full and marginal
# p12 = p1*p2

##

selectedLabels=Vector{Vector{Int}}()
p12 = manifoldProduct([p1; p2]; addEntropy=false,
                                recordLabels=true,
                                selectedLabels=selectedLabels  )
##

p12_ = marginal(p12, [1;2])

## intermediate test, check product of selected kernels match what is in the marginal



##

# now do submanifold dimensions separately as reference test -- should get a similar result
pts1_ = getPoints(p1_)
pts2_ = getPoints(p2)

p1__ = manikde!(TranslationGroup(2), pts1_)
p2__ = manikde!(TranslationGroup(2), pts2_)

# p12__ = p1__*p2__
selectedLabels__=Vector{Vector{Int}}()
p12__ = manifoldProduct([p1__; p2__]; addEntropy=true,
                                      recordLabels=true,
                                      selectedLabels=selectedLabels__  )
#

##

_p12_ = manikde!(TranslationGroup(2), getPoints(p12_))

@test mmd(_p12_, p12__) < 0.001



## Plots showing the problem, p12 is wrong!!
# using Cairo, RoMEPlotting
# Gadfly.set_default_plot_size(35cm,20cm)

# plotKDE([p1_;p2; p12], levels=1, selectedPoints=selectedLabels[1:2])
# n=3; plotKDE([p1__;p2__; p12__], levels=1, selectedPoints=selectedLabels__[n:n])
# plotKDE([p1__; p2__; p12__])
# plotKDE([_p12_; p12__])
##
end

#