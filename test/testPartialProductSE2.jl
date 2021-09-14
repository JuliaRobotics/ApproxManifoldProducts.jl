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

# # p1 full SpecialEuclidean(2)
# pts1 = [exp(M, e0, hat(M, e0, [randn(2); 0.1*randn()])) for _ in 1:100]
p1 = manikde!(M, pts1)
p1_ = marginal(p1,[1;2])

# # p2 only Translation(2) part
# pts2 = vcat([exp(M, e0, hat(M, e0, [5;0;pi/4] .+ [5*randn(2);0.1*randn()])) for _ in 1:50], [exp(M, e0, hat(M, e0, [5;-5;pi/4] .+ [10*randn(2);0.1*randn()])) for _ in 1:50])
# shuffle!(pts2)
_p2 = manikde!(M, pts2)
p2 = marginal(_p2, [1;2])

# product of full and marginal
p12 = p1*p2

p12_ = marginal(p12, [1;2])


# now do submanifold dimensions separately as reference test -- should get a similar result
pts1_ = getPoints(p1_)
pts2_ = getPoints(p2)

p1__ = manikde!(TranslationGroup(2), pts1_)
p2__ = manikde!(TranslationGroup(2), pts2_)

p12__ = p1__*p2__

_p12_ = manikde!(TranslationGroup(2), getPoints(p12_))

@test mmd(_p12_, p12__) < 0.001



## Plots showing the problem, p12 is wrong!!
# using Cairo, RoMEPlotting
# Gadfly.set_default_plot_size(35cm,20cm)

# plotKDE([p1_;p2; p12], levels=5)
# plotKDE([p1__; p2__; p12__])
# plotKDE([_p12_; p12__])
##
end

#