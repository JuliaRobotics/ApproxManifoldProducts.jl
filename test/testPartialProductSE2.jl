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

len = length(pts1)

# define test manifold
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

@test !isPartial(p12)

p12_ = marginal(p12, [1;2])

@test isPartial(p12_)

## intermediate test, check product of selected kernels match what is in the marginal

for sidx = 1:len
  bw1 = getBW(p1)[:,1] .^2
  bw2 = getBW(p2, false)[:,1] .^2

  u1 = pts1[selectedLabels[sidx][1]]
  u2 = pts2[selectedLabels[sidx][2]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);

  @test isapprox( u12.parts[1], getPoints(p12)[sidx].parts[1] )
end

## now check the marginal dimensions only

# now do submanifold dimensions separately as reference test -- should get a similar result
pts1_ = getPoints(p1_)
pts2_ = getPoints(p2)

## Do the translation part separate

for sidx = 1:len
  bw1 = getBW(p1_)[:,1] .^2
  bw2 = getBW(p2)[:,1] .^2

  u1 = pts1_[selectedLabels[sidx][1]]
  u2 = pts2_[selectedLabels[sidx][2]]

  u12, = calcProductGaussians(TranslationGroup(2), [u1,u2], [bw1,bw2])

  @test isapprox( u12, getPoints(p12)[sidx].parts[1] )
end

## NEW PRODUCT ON ONLY THE TRANSLATION PART, COMPARE WITH ABOVE RESULT

p1__ = manikde!(TranslationGroup(2), pts1_)
p2__ = manikde!(TranslationGroup(2), pts2_)


# p12__ = p1__*p2__
selectedLabels__=Vector{Vector{Int}}()
p12__ = manifoldProduct([p1__; p2__]; addEntropy=false,
                                      recordLabels=true,
                                      selectedLabels=selectedLabels__  )
#

##

sidx = 1
for sidx = 1:len

bw1 = getBW(p1__)[:,1] .^2
bw2 = getBW(p2__)[:,1] .^2

u1 = pts1_[selectedLabels__[sidx][1]]
u2 = pts2_[selectedLabels__[sidx][2]]

u12, = calcProductGaussians(TranslationGroup(2), [u1,u2], [bw1,bw2])

# @test isapprox( u12, getPoints(p12)[sidx].parts[1] )

end

## check calcProduct 


sidx = 1

bw1 = getBW(p1)[:,1] .^2
bw2 = getBW(p2, false)[:,1] .^2

u1 = pts1[selectedLabels__[sidx][1]]
u2 = pts2[selectedLabels__[sidx][2]]

u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);



##


_p12_ = manikde!(TranslationGroup(2), getPoints(p12_))


@test_broken mmd(_p12_, p12__) < 0.001



# ## Plots showing the problem, p12 is wrong!!
# using Cairo, RoMEPlotting
# Gadfly.set_default_plot_size(35cm,20cm)

# n=1; plotKDE([p1_;p2; p12], levels=3, selectedPoints=selectedLabels[n:n])
# n=2; plotKDE([p1__;p2__; p12__], levels=3, selectedPoints=selectedLabels__[n:n])
# plotKDE([p1__; p2__; p12__])

# plotKDE([_p12_; p12__])
##
end

#