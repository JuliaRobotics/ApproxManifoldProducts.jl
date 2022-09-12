
using Test
using ApproxManifoldProducts
using StaticArrays, Manifolds

##

@testset "Trivial ManifoldBalancedBallTree case on Euclidean(1)" begin
##  

M = Euclidean(1)

pts = [
  -1.0
  1.0
]

mbb = AMP.ManifoldBalancedBallTree(M, pts; leafsize=1, reorder=true)



##
end

@testset "Test specific ManifoldBalancedBallTree case on Euclidean(1)" begin
##  


M = Euclidean(1)

pts = [
  1.6428412203258511
  -0.4823265406855113
  0.4354221188230193
  1.908228908562008
  0.9791603637197599
  1.0798652993450037
  1.0875113872287496
  1.2019334066681153
  0.013282018302654335
  0.965302261228411   
]

ref = AMP.NNR.BallTree(reshape(pts,1,:), AMP.DST.Euclidean(); leafsize=2)

mbb = AMP.ManifoldBalancedBallTree(M, pts; leafsize=1, reorder=true)


# mbb.hyper_spheres



##
end


@testset "Test ManifoldBalancedBallTree on Euclidean(2)" begin
##

M = Euclidean(2)
pts = [ SA[randn();randn()] for _ in 1:10 ]


mbb = AMP.ManifoldBalancedBallTree(M, pts)

##
end


@testset "Test ManifoldBalancedBallTree on SpecialOrthogonal(2)" begin
##

M = SpecialOrthogonal(2)
pts = [ SMatrix{2,2}(exp_lie(M, hat(M, Identity(M), randn()))) for _ in 1:10 ]

Base.@kwdef struct SO2Dist{M_} <: AMP.DST.Metric
  M::M_ = SpecialOrthogonal(2)
end

(d::SO2Dist)(p::AbstractMatrix,q::AbstractMatrix) = Manifolds.ManifoldsBase.distance(d.M,p,q)

mbb = AMP.ManifoldBalancedBallTree(M, pts, SO2Dist(); leafsize=1)




##
end

#