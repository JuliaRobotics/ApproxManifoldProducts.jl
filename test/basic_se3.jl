# test SE3 kernel and MMD

using ApproxManifoldProducts
using Test


@testset "Basic SE3 manifold test (transition version)" begin

a = randn(6,50)
b = randn(6,50)

ret = mmd(a, b, SE3_Manifold)

@test ret < 50

end


@testset "Test isapprox function on basic SE3" begin

# FIXME hacking the manifolds here, needs consolidation
a = kde!(randn(6,50))
b = kde!(randn(6,50))
c_ = randn(6,50)
c_[1,:] .+= 50
c = kde!(c_)

A = ManifoldKernelDensity(SE3_Manifold, a)
B = ManifoldKernelDensity(SE3_Manifold, b)
C = ManifoldKernelDensity(SE3_Manifold, c)

@test isapprox(A, B)
@test !isapprox(A, C)

end


#
