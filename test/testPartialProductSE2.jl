# test on partial products with SpecialEuclidean(2; vectors=HybridTangentRepresentation())

using Manifolds
using ApproxManifoldProducts
using Test
using BSON

##

@testset "partial product with a SpecialEuclidean(2; vectors=HybridTangentRepresentation())" begin
##

datafile = joinpath(@__DIR__, "testdata", "partialtest.bson")
# BSON.save(datafile, dict)
data = BSON.load(datafile)
pts1 = data[:dict][:pts1]
pts2 = data[:dict][:pts2]

randU = Float64[]
randN = Float64[]

len = length(pts1)

# define test manifold
M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
e0 = ArrayPartition([0.0;0.0], [1 0; 0 1.0]) # identity_element(M)

# p1 full SpecialEuclidean(2; vectors=HybridTangentRepresentation())
p1 = manikde!(M, pts1)
p1_ = marginal(p1,[1;2])

# p2 only Translation(2) part
_p2 = manikde!(M, pts2)
p2 = marginal(_p2, [1;2])

# product of full and marginal
# p12 = p1*p2

## =======================FIRST PRODUCT================================================

selectedLabels=Vector{Vector{Int}}()
p12 = manifoldProduct([p1; p2]; addEntropy=false,
                                recordLabels=true,
                                selectedLabels=selectedLabels,
                                _randU=randU,
                                _randN=randN  )
#
selectedLabels
#

@test !isPartial(p12)

p12_ = marginal(p12, [1;2])
@test isPartial(p12_)

_p12_ = manikde!(TranslationGroup(2), getPoints(p12_))

## intermediate test, check product of selected kernels match what is in the marginal

for sidx = 1:len
  bw1 = getBW(p1)[:,1] .^2
  bw2 = getBW(p2, false)[:,1] .^2

  u1 = pts1[selectedLabels[sidx][1]]
  u2 = pts2[selectedLabels[sidx][2]]

  u12 = calcProductGaussians(M, [u1,u2], [bw1,bw2])

  @test_broken isapprox( submanifold_component(mean(u12),1), submanifold_component(getPoints(p12)[sidx],1) )
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

  u12 = calcProductGaussians(TranslationGroup(2), [u1,u2], [bw1,bw2])

  @test isapprox( mean(u12), submanifold_component(getPoints(p12)[sidx],1) )
end



## =======================SECOND PRODUCT================================================
## NEW PRODUCT ON ONLY THE TRANSLATION PART, COMPARE WITH ABOVE RESULT


p1__ = manikde!(TranslationGroup(2), pts1_)
p2__ = manikde!(TranslationGroup(2), pts2_)

# p12__ = p1__*p2__
selectedLabels__=Vector{Vector{Int}}()
p12__ = manifoldProduct([p1__; p2__]; addEntropy=false,
                                      recordLabels=true,
                                      selectedLabels=selectedLabels__,
                                      _randU=randU,
                                      _randN=randN  )
#
selectedLabels__


## compare calcProduct of full*partial with selection from partial*partial



sidx = 1
for sidx = 1:len

bw1 = getBW(p1)[:,1] .^2
bw2 = getBW(p2, false)[:,1] .^2

# full-partial points, but selected from partial-partial product
u1 = pts1[selectedLabels__[sidx][1]]
u2 = pts2[selectedLabels__[sidx][2]]

u12 = calcProductGaussians(M, [u1,u2], [bw1,bw2]);
u12_ = calcProductGaussians(TranslationGroup(2), [submanifold_component(u1,1),submanifold_component(u2,1)], [bw1[1:2],bw2[1:2]]);

@test_broken isapprox( submanifold_component(mean(u12),1), mean(u12_); atol=0.001 ) # atol = 0.1
@test isapprox( getPoints(p12__)[sidx], mean(u12_) )

end

##


@test mmd(_p12_, p12__) < 0.001


##
end



## DEBUG PLOTTING ==========================================================================

## Plots showing the problem, p12 was wrong!!

# using Cairo, RoMEPlotting
# Gadfly.set_default_plot_size(35cm,20cm)

# n=10; plotKDE([p1_;p2; p12], levels=3, selectedPoints=selectedLabels[n:n])
# n=3; plotKDE([p1__;p2__; p12__], levels=3, selectedPoints=selectedLabels__[n:n])
# plotKDE([p1__; p2__; p12__])

# plotKDE([_p12_; p12__])


##