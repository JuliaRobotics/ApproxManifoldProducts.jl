# test SE3 kernel and MMD

using ApproxManifoldProducts
using Test


@testset "Basic SE3 manifold test (transition version)" begin

a = randn(6,50)
b = randn(6,50)

ret = mmd(a, b, SE3_Manifold)

@test ret < 50

end




#
