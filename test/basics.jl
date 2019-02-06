
## Great

using ApproxManifoldProducts
using Test
using LinearAlgebra

using KernelDensityEstimate


# pts = [-pi+0.15; -pi+0.2; +pi-0.2; pi-0.149]
#
# pcm = kde!(pts, [0.05], (addtheta,), (difftheta,) )
#
#
#
# pcm.bt.centers



## but wait

# TODO ensure this manifold product is still valid
# @testset "basic tests..." begin
#
#
# mbe1 = ManifoldBelief(EuclideanManifold, 0.0)
# mbe2 = ManifoldBelief(EuclideanManifold, 0.0)
#
# *([mbe1;mbe2])
#
# end

##

@testset "test circular difftheta..." begin

th1 = -0.1
th2 = +0.1

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10

th1 = +0.1
th2 = -0.1

# should result in -0.2
@test -1e-10 < difftheta(th2, th1) + 0.2 < 1e-10
# should result in +0.2
@test -1e-10 < difftheta(th1, th2) - 0.2 < 1e-10

th1 = pi-0.1
th2 = -pi+0.1

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10

th1 = pi-0.1 + 2pi
th2 = -pi+0.1 +2pi

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10

th1 = pi-0.1  -2pi
th2 = -pi+0.1 -2pi

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10

th1 = pi-0.1  +2pi
th2 = -pi+0.1 -2pi

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10


th1 = pi-0.1  -2pi
th2 = -pi+0.1 +2pi

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10


th1 = -0.1 + pi/2.0
th2 = +0.1 + pi/2.0

# should result in +0.2
@test -1e-10 < difftheta(th2, th1) - 0.2 < 1e-10
# should result in -0.2
@test -1e-10 < difftheta(th1, th2) + 0.2 < 1e-10


eps_l = 1e-5
th1 = 0.0
th2 = -pi + eps_l

@test 0.0 < difftheta(th2, th1) + pi < 2*eps_l
@test -2*eps_l < difftheta(th1, th2) - pi < 0.0

end


##


@testset "test circular manifold BallTree construction small positive..." begin

pts = [0.15; 0.2; 0.35; 0.4]

p = kde!(pts, [0.05])

pcp = kde!(pts, [0.05], (addtheta,) )

pcm = kde!(pts, [0.05], (addtheta,), (difftheta,) )

@test norm(p.bt.centers - pcp.bt.centers) < 1e-10
@test norm(p.bt.centers - pcm.bt.centers) < 1e-10

@test norm(p.bt.ranges - pcp.bt.ranges) < 1e-10
@test norm(p.bt.ranges - pcm.bt.ranges) < 1e-10

@test norm(p.bt.weights - pcp.bt.weights) < 1e-10
@test norm(p.bt.weights - pcm.bt.weights) < 1e-10

@test norm(p.bt.left_child - pcp.bt.left_child) < 1e-10
@test norm(p.bt.left_child - pcm.bt.left_child) < 1e-10

@test norm(p.bt.right_child - pcp.bt.right_child) < 1e-10
@test norm(p.bt.right_child - pcm.bt.right_child) < 1e-10

@test norm(p.bt.lowest_leaf - pcp.bt.lowest_leaf) < 1e-10
@test norm(p.bt.lowest_leaf - pcm.bt.lowest_leaf) < 1e-10

@test norm(p.bt.highest_leaf - pcp.bt.highest_leaf) < 1e-10
@test norm(p.bt.highest_leaf - pcm.bt.highest_leaf) < 1e-10

@test norm(p.bt.permutation - pcp.bt.permutation) < 1e-10
@test norm(p.bt.permutation - pcm.bt.permutation) < 1e-10

end

##


@testset "test circular manifold BallTree construction small negative..." begin

pts = -[0.15; 0.2; 0.35; 0.4]

p = kde!(pts, [0.05])

pcp = kde!(pts, [0.05], (addtheta,) )

pcm = kde!(pts, [0.05], (addtheta,), (difftheta,) )

# p.bt.centers
@test norm(p.bt.centers - pcp.bt.centers) < 1e-10
@test norm(p.bt.centers - pcm.bt.centers) < 1e-10

@test norm(p.bt.ranges - pcp.bt.ranges) < 1e-10
@test norm(p.bt.ranges - pcm.bt.ranges) < 1e-10

@test norm(p.bt.weights - pcp.bt.weights) < 1e-10
@test norm(p.bt.weights - pcm.bt.weights) < 1e-10

@test norm(p.bt.left_child - pcp.bt.left_child) < 1e-10
@test norm(p.bt.left_child - pcm.bt.left_child) < 1e-10

@test norm(p.bt.right_child - pcp.bt.right_child) < 1e-10
@test norm(p.bt.right_child - pcm.bt.right_child) < 1e-10

@test norm(p.bt.lowest_leaf - pcp.bt.lowest_leaf) < 1e-10
@test norm(p.bt.lowest_leaf - pcm.bt.lowest_leaf) < 1e-10

@test norm(p.bt.highest_leaf - pcp.bt.highest_leaf) < 1e-10
@test norm(p.bt.highest_leaf - pcm.bt.highest_leaf) < 1e-10

@test norm(p.bt.permutation - pcp.bt.permutation) < 1e-10
@test norm(p.bt.permutation - pcm.bt.permutation) < 1e-10

end



##


@testset "test circular manifold BallTree construction around 0rad..." begin

pts = [-0.2; 0.15; 0.15; 0.2]

p = kde!(pts, [0.05])

pcp = kde!(pts, [0.05], (addtheta,) )

pcm = kde!(pts, [0.05], (addtheta,), (difftheta,) )

# p.bt.centers
@test norm(p.bt.centers - pcp.bt.centers) < 1e-10
@test norm(p.bt.centers - pcm.bt.centers) < 1e-10

@test norm(p.bt.ranges - pcp.bt.ranges) < 1e-10
@test norm(p.bt.ranges - pcm.bt.ranges) < 1e-10

@test norm(p.bt.weights - pcp.bt.weights) < 1e-10
@test norm(p.bt.weights - pcm.bt.weights) < 1e-10

@test norm(p.bt.left_child - pcp.bt.left_child) < 1e-10
@test norm(p.bt.left_child - pcm.bt.left_child) < 1e-10

@test norm(p.bt.right_child - pcp.bt.right_child) < 1e-10
@test norm(p.bt.right_child - pcm.bt.right_child) < 1e-10

@test norm(p.bt.lowest_leaf - pcp.bt.lowest_leaf) < 1e-10
@test norm(p.bt.lowest_leaf - pcm.bt.lowest_leaf) < 1e-10

@test norm(p.bt.highest_leaf - pcp.bt.highest_leaf) < 1e-10
@test norm(p.bt.highest_leaf - pcm.bt.highest_leaf) < 1e-10

@test norm(p.bt.permutation - pcp.bt.permutation) < 1e-10
@test norm(p.bt.permutation - pcm.bt.permutation) < 1e-10

end


##


@testset "test circular manifold BallTree construction around π wrap point..." begin

# pts = [pi-0.2; pi-0.15; -pi+0.149; -pi+0.2]
pts = [-pi+0.15; -pi+0.2; +pi-0.2; pi-0.149]

# p = kde!(pts, [0.05])

# pcp = kde!(pts, [0.05], (addtheta,) )

pcm = kde!(pts, [0.05], (addtheta,), (difftheta,) )

@test abs(pcm.bt.centers[1] + pi) < 1e-3

@test abs(pcm.bt.centers[2] - pi + 0.175) < 1e-3
@test abs(pcm.bt.centers[3] + pi - 0.175) < 1e-3


@test abs(pcm.bt.centers[5] - pts[3]) < 1e-3
@test abs(pcm.bt.centers[6] - pts[4]) < 1e-3
@test abs(pcm.bt.centers[7] - pts[1]) < 1e-3
@test abs(pcm.bt.centers[8] - pts[2]) < 1e-3



end
