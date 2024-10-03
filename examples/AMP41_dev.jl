# dev testing AMP 41

using Revise
using StaticArrays, Manifolds, NearestNeighbors, Distances

import NearestNeighbors: interpolate

##



# convert point to coordinates
function coords(p)
    return SA[p.x[1][1], p.x[1][2], atan(p.x[2][2,1],p.x[2][1,1])] # fixed from acos(p.parts[2][1,1]
end
# reverse of `coords`
function uncoords(p)
    α = p[3]
    return ArrayPartition(SA[p[1], p[2]], SA[cos(α) -sin(α); sin(α) cos(α)])
end

# The variant in ManifoldML doesn't support `ProductRepr` currently.
struct SE2Distance{TM<:AbstractManifold} <: Distances.Metric
    manifold::TM
end
function (dist::SE2Distance)(a, b)
    return distance(dist.manifold, uncoords(a), uncoords(b))
end

##

M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
N = 128


# some random points to make a tree from
pts = [uncoords(@SVector randn(3)) for _ in 1:N]


dist = SE2Distance(M)
# making a tree
point_matrix = reduce(hcat, map(a -> coords(a), pts))
balltree = BallTree(point_matrix, dist; leafsize=1)
# finding nearest neighbors
k = 3
idxs, dists = knn(balltree, coords(pts[2]), k)



## ========================




import ApproxManifoldProducts: HyperEllipse, ManellicTree, eigenCoords, splitPointsEigen





function buildTree(
  tree,
  pts
)
  #




1 == length(pts) ? nothing : nothing


end




## ====================================================================
## TESTING CODE
## ====================================================================



