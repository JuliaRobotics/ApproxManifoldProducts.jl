# test basic manifold product behaviour

using ApproxManifoldProducts
using KernelDensityEstimate
using Test


# # 2D
#
# pts1 = [0.05*randn(100)'; 0.5*randn(100)']
# pts2 = [0.05*randn(100)'; 0.5*randn(100)']
#
# P1 = manikde!(pts1, (:Euclid, :Circular))
# P2 = manikde!(pts2, (:Euclid, :Circular))
#
# P12 = P1*P2
#
#
#
# plotKDE([P1;P2;P12], c=["red";"blue";"magenta"],levels=1)
#


# 3D

@testset "Basic 3D multiply and cross dimension covariance test..." begin

N = 100

pts1 = [0.05*randn(N)'; 0.05*randn(N)'; 0.75*randn(N)']
pts2 = [0.05*randn(N)'; 0.05*randn(N)'; 0.75*randn(N)']

# P1 = manikde!(pts1, (:Euclid,:Euclid,:Circular))
# P2 = manikde!(pts2, (:Euclid,:Euclid,:Circular))

# P1 = manikde!(pts1, (:Euclid,:Euclid,:Euclid))
# P2 = manikde!(pts2, (:Euclid,:Euclid,:Euclid))

P2 = kde!(pts2)
P1 = kde!(pts1)

P12 = P1*P2

pts = getPoints(P12)

@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)




P1 = manikde!(pts1, (:Euclid,:Euclid,:Euclid))
P2 = manikde!(pts2, (:Euclid,:Euclid,:Euclid))

P12 = P1*P2

pts = getPoints(P12)

@test 0.8*N < sum(abs.(pts[1,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[2,:]) .< 0.1)
@test 0.8*N < sum(abs.(pts[3,:]) .< 2.0)





P1 = manikde!(pts1, (:Euclid,:Euclid,:Circular))
P2 = manikde!(pts2, (:Euclid,:Euclid,:Circular))

P12 = P1*P2

pts = getPoints(P12)

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
