using ApproxManifoldProducts
using Test
using LinearAlgebra

@testset "basic tests..." begin


mbe1 = ManifoldBelief(EuclideanManifold, 0.0)
mbe2 = ManifoldBelief(EuclideanManifold, 0.0)

*([mbe1;mbe2])



end
