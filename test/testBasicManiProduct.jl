# test basic manifold product behaviour

using ApproxManifoldProducts
using KernelDensityEstimate
using Test
using TensorCast
using Manifolds


## 3D

@testset "Basic 3D multiply and cross dimension covariance test..." begin

##

N = 100

pts1 = [[0.05*randn(2);0.75*randn()] for i in 1:N]
pts2 = [[0.05*randn(2);0.75*randn()] for i in 1:N]


P1 = manikde!(Euclidean(3),pts1)
P2 = manikde!(Euclidean(3),pts2)

P12 = P1*P2

@test typeof(P12._u0) <: Vector{Float64}

pts_ = getPoints(P12)

pts = AMP._pointsToMatrixCoords(P12.manifold, pts_)

@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)

##

M = SpecialEuclidean(2)
u0 = ProductRepr(zeros(2),[1 0; 0 1.0])
ϵ = identity(M, u0)

pts1 = [exp(M, ϵ, hat(M, ϵ, [0.05*randn(2);0.75*randn()])) for i in 1:N]
pts2 = [exp(M, ϵ, hat(M, ϵ, [0.05*randn(2);0.75*randn()])) for i in 1:N]

P1 = manikde!(M, pts1)
P2 = manikde!(M, pts2)

P12 = P1*P2

pts_ = getPoints(P12)

pts = AMP._pointsToMatrixCoords(P12.manifold, pts_)


@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)


# using KernelDensityEstimatePlotting
# using Gadfly
# Gadfly.set_default_plot_size(35cm,25cm)
#
# plotKDE([P1;P2;P12], c=["red";"blue";"magenta"],levels=1) |> PDF("/tmp/test.pdf")
# run(`evince /tmp/test.pdf`)

##

end



#
