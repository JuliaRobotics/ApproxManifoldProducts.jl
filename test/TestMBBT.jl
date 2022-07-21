
using Test
using ApproxManifoldProducts
using StaticArrays, Manifolds

##


@testset "Test ManifoldBalancedBallTree on SpecialOrthogonal(2)" begin
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