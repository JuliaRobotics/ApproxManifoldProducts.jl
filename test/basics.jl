using ApproxManifoldProducts
using Test
using LinearAlgebra

@testset "basic tests..." begin


mbe1 = ManifoldBelief(EuclideanManifold, 0.0)
mbe2 = ManifoldBelief(EuclideanManifold, 0.0)

*([mbe1;mbe2])

mbr1 = ManifoldBelief(SO2Manifold, 0.0)
mbr2 = ManifoldBelief(SO2Manifold, 0.0)

*([mbr1;mbr2])


end
