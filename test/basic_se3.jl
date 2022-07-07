# test SE3 kernel and MMD

using ApproxManifoldProducts
using Manifolds
using Test

##

@testset "Test isapprox function on basic SpecialEuclidean(3)" begin

##

# FIXME hacking the manifolds here, needs consolidation
# a = kde!(randn(6,50))
# b = kde!(randn(6,50))
# c_ = randn(6,50)
# c_[1,:] .+= 50
# c = kde!(c_)

M = SpecialEuclidean(3)
u0 = ArrayPartition(zeros(3),[1 0 0; 0 1 0; 0 0 1.0])
ϵ = identity_element(M, u0)
N = 50

pts1 = [exp(M, ϵ, hat(M, ϵ, [0.5*randn(3);0.1*randn(3)])) for i in 1:N]
pts2 = [exp(M, ϵ, hat(M, ϵ, [0.5*randn(3);0.1*randn(3)])) for i in 1:N]
pts3 = [exp(M, ϵ, hat(M, ϵ, [0.5*randn()+50;0.5*randn(2);0.1*randn(3)])) for i in 1:N]

##

ret = mmd(M, pts1, pts2)
@test ret < 1

ret = mmd(M, pts1, pts3)
@test 0.001 < ret


##

A = ManifoldKernelDensity(M, pts1)
B = ManifoldKernelDensity(M, pts2)
C = ManifoldKernelDensity(M, pts3)

@test isapprox(A, B)
@test !isapprox(A, C)

##

show(A)

##

end


#
