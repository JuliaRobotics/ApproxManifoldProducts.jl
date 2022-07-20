

# using Revise

using Manifolds, NearestNeighbors, Distances, StaticArrays
import Manifolds: ArrayPartition

##

Base.@kwdef struct DistSE2 <: Distances.Metric
  M = SpecialEuclidean(2)
end 
# (d::DistSE2)(p,q) = Manifolds.ManifoldsBase.distance(d.M,p,q)
(d::DistSE2)(p::ArrayPartition,q::ArrayPartition) = Manifolds.ManifoldsBase.distance(d.M,p,q)
function (d::DistSE2)(p::ArrayPartition,q_::AbstractVector)
  q = ArrayPartition(SA[q_[1],q_[2]], SMatrix{2,2}(q_[3],q_[4],q_[5],q_[6]))
  Manifolds.ManifoldsBase.distance(d.M,p,q)
end
function (d::DistSE2)(p_::AbstractVector,q_::AbstractVector)
  p = ArrayPartition(SA[p_[1],p_[2]], SMatrix{2,2}(p_[3],p_[4],p_[5],p_[6]))
  q = ArrayPartition(SA[q_[1],q_[2]], SMatrix{2,2}(q_[3],q_[4],q_[5],q_[6]))
  Manifolds.ManifoldsBase.distance(d.M,p,q)
end

# see https://github.com/SciML/RecursiveArrayTools.jl/pull/220
Base.length(::Type{<:ArrayPartition{F,T}}) where {F,N,T <: NTuple{N,StaticArray}} = T.parameters .|> length |> sum

##

pts = [ArrayPartition(SA[randn(2)...], SMatrix{2,2}([1 0; 0 1.])) for _ in 1:10]
dSE2 = DistSE2()

##

bt = NearestNeighbors.BallTree(pts, dSE2; leafsize=1)

## How to get any node in the BallTree



##