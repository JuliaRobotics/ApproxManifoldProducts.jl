# test SE3 kernel and MMD

using ApproxManifoldProducts
using Test


@testset "Basic SE3 manifold test (transition version)" begin

a = randn(6,50)
b = randn(6,50)

ret = mmd(SE3_Manifold, a, b)

@test ret < 50


end




#
