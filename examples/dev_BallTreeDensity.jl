## mateusz recommendation Manifolds.jl ProductRepr, see AMP.jl#41


using StaticArrays, Manifolds, NearestNeighbors, Distances

M = SpecialEuclidean(2)
N = 100
# convert point to coordinates
function coords(p)
    return SA[p.parts[1][1], p.parts[1][2], acos(p.parts[2][1,1])]
end
# reverse of `coords`
function uncoords(p)
    α = p[3]
    return ProductRepr((SA[p[1], p[2]]), SA[cos(α) -sin(α); sin(α) cos(α)])
end
# some random points to make a tree from
pts = [uncoords(@SVector randn(3)) for _ in 1:N]

# The variant in ManifoldML doesn't support `ProductRepr` currently.
struct SE2Distance{TM<:Manifold} <: Distances.Metric
    manifold::TM
end
function (dist::SE2Distance)(a, b)
    return distance(dist.manifold, uncoords(a), uncoords(b))
end

dist = SE2Distance(M)
# making a tree
vector_elem = coords.(pts)
balltree = BallTree(vector_elem, dist)

# point_matrix = reduce(hcat, map(a -> coords(a), pts))
# balltree = BallTree(point_matrix, dist)
# finding nearest neighbors
k = 3
idxs, dists = knn(balltree, coords(pts[2]), k)













## build BallTree for KDE density estimation


using NearestNeighbors: BallTree, RMSDeviation
using Colors
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

using StaticArrays
using LinearAlgebra

using BenchmarkTools


##

allpts = [SVector(randn(2)...) for i in 1:100];

tree = BallTree(allpts, RMSDeviation(); leafsize = 50)

tree = BallTree(allpts, RMSDeviation(); leafsize = 25)

tree = BallTree(allpts, RMSDeviation(); leafsize = 13)

tree = BallTree(allpts, RMSDeviation(); leafsize = 7)

tree = BallTree(rand(2,100), RMSDeviation(); leafsize = 1)



## Make own distance

# using TransformUtils
using Manifolds

# carfully only include a single definition for use with NearestNeighbors.jl
using Distances: Metric


struct ManiRMSDeviation{M <: Manifold} <: Metric where M
  manifold::M
end

##



mr = ManiRMSDeviation(Euclidean(2))



##



distance(mr.manifold, 1,2)


##

distance(mr.manifold, SVector(1,2),SVector(2,3))





##


M_se2 = SpecialEuclidean(2)


sI = SMatrix{3,3,Float64}(diagm(ones(3)))

T1 = Manifolds.hat(M_se2, sI, SVector(0,0,pi/2))

T2 = Manifolds.hat(M_se2, sI, SVector(1,0,0))


distance(M_se2, T1, T2)




##


md = ManiRMSDeviation{}()

rd = RMSDeviation()

isconcretetype(rd)

# (dist::MeanSqDeviation)(a, b) = sqeuclidean(a, b) / length(a)
# msd(a, b) = MeanSqDeviation()(a, b)

# (dist::RMSDeviation)(a, b) = sqrt(MeanSqDeviation()(a, b))
# rmsd(a, b) = RMSDeviation()(a, b)


@code_warntype rmsd(1,1)



sqeuclidean(a, b) / length(a)

(::ManiRMSDeviation{T})(a, b) where T = sqrt(MeanSqDeviation()(a, b))




##

import NearestNeighbors.HyperSphere

##

# Adds a sphere to an axis
function add_sphere(ax, hs::HyperSphere, col)
    ell = patch.Circle(hs.center, radius = hs.r, facecolor="none", edgecolor=col)
    ax.add_artist(ell)
end

# Skip non leaf nodes
offset = tree.tree_data.n_internal_nodes + 1
nleafs = tree.tree_data.n_leafs

# Range of leaf nodes
index_range = offset: offset + nleafs - 1

# Generate some nice colors
cols = distinguishable_colors(length(index_range), RGB(0,0,0))

# Create figure
cfig = figure()
ax = cfig.add_subplot(1,1,1)
ax.set_aspect("equal")
axis((-.25,1.25,-.25,1.25))


for (i, idx) = enumerate(index_range)
  col = cols[i]
  # Get the indices of the leaf nodes into the tree data
  range = NearestNeighbors.get_leaf_range(tree.tree_data, idx)
  d = tree.data[range]
  
  # Plot the points in the hyper spehre
  plot(getindex.(d, 1), getindex.(d, 2), "*", color = (col.r, col.g, col.b))
  
  # And the hypersphere itself
  sphere = tree.hyper_spheres[idx]
  add_sphere(ax, sphere, (col.r, col.g, col.b))
end

title("Leaf nodes with their corresponding points")




##

using Distances


MeanSqDeviation()([1;2;3;4], [2;3;4;5])




#