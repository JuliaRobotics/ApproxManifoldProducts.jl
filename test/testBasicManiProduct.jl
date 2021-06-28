# test basic manifold product behaviour

using ApproxManifoldProducts
using KernelDensityEstimate
using Test
using TensorCast
using Manifolds


# 3D

@testset "Basic 3D multiply and cross dimension covariance test..." begin

N = 100

pts1_ = [0.05*randn(N)'; 0.05*randn(N)'; 0.75*randn(N)']
pts2_ = [0.05*randn(N)'; 0.05*randn(N)'; 0.75*randn(N)']

# lazy
@cast pts1[j][i] := pts1_[i,j]
@cast pts2[j][i] := pts2_[i,j]


P1 = manikde!(pts1, Euclidean(3))
P2 = manikde!(pts2, Euclidean(3))

P12 = P1*P2

pts_ = getPoints(P12)

@cast pts[i,j] := pts_[j][i]

@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)



P1 = manikde!(pts1, SpecialEuclidean(2))
P2 = manikde!(pts2, SpecialEuclidean(2))

P12 = P1*P2

pts_ = getPoints(P12)

@cast pts[i,j] := pts_[j][i]

@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)



# using KernelDensityEstimatePlotting
# using Gadfly
# Gadfly.set_default_plot_size(35cm,25cm)
#
# plotKDE([P1;P2;P12], c=["red";"blue";"magenta"],levels=1) |> PDF("/tmp/test.pdf")
# run(`evince /tmp/test.pdf`)

end



#
