# The NearestNeighbors.jl package is licensed under the MIT "Expat" License:

# > Copyright (c) 2015: Kristoffer Carlsson.
# > Copyright (c) 2022: Dehann Fourie
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


using Manifolds, StaticArrays

import Distances as DST
# import Distances: Metric
import NearestNeighbors: TreeData, NNTree
import Manifolds: ArrayPartition


# A ManifoldBallTreeBalanced (also called Metric tree) is a tree that is created
# from successively splitting points into surrounding hyper spheres
# which radius are determined from the given metric.
# The tree uses the triangle inequality to prune the search space
# when finding the neighbors to a point,
struct ManifoldBallTreeBalanced{V <: ArrayPartition,T,M <: DST.Metric,T} <: NNTree{V,M}
    """ tree points exist on some manifold `<:Manifolds.AbstractManifold` """
    manifold::T
    """ data points associated with this tree """
    data::Vector{V}
    """ Each hyper sphere bounds its children """
    hyper_spheres::Vector{ManifoldHyperSphere{V,T}} 
    """ Translates from tree index -> point index """
    indices::Vector{Int}                            
    """ Metric used for tree """
    metric::M                                       
    """ Some constants needed """
    tree_data::TreeDataBalanced                     
    """ If the data has been reordered """
    reordered::Bool                                 
end

# When we create the bounding spheres we need some temporary arrays.
# We create a type to hold them to not allocate these arrays at every
# function call and to reduce the number of parameters in the tree builder.
Base.@kwdef struct ArrayPartitionBuffers{N,T <: ArrayPartition}
    center::MVector{N,T} = MVector{N,T}(undef)
end

function ArrayPartitionBuffers(::Type{Val{N}}, ::Type{T}) where {N, T}
    ArrayPartitionBuffers{N,T}()
end


"""
    ManifoldBallTreeBalanced(data [, metric = Euclidean(); leafsize = 10, reorder = true]) -> ManifoldballtreeBalanced

Creates a `ManifoldBallTreeBalanced` from the data using the given `metric` and `leafsize`.
"""
function ManifoldBallTreeBalanced(
                    mani::AbstractManifold,
                    data::AbstractVector{V},
                    metric::M = Euclidean();
                    leafsize::Int = 10,
                    reorder::Bool = true,
                    storedata::Bool = true,
                    reorderbuffer::Vector{V} = Vector{V}()) where {V <: AbstractArray, M <: DST.Metric}
    #
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    tree_data = TreeData(data, leafsize)
    n_d = length(V)
    n_p = length(data)

    array_buffs = ArrayPartitionBuffers(Val{length(V)}, get_T(eltype(V)))
    indices = collect(1:n_p)

    # Bottom up creation of hyper spheres so need spheres even for leafs)
    hyper_spheres = Vector{ManifoldHyperSphere{length(V),eltype(V)}}(undef, tree_data.n_internal_nodes + tree_data.n_leafs)

    if reorder
        indices_reordered = Vector{Int}(undef, n_p)
        if isempty(reorderbuffer)
            data_reordered = Vector{V}(undef, n_p)
        else
            data_reordered = reorderbuffer
        end
    else
        # Dummy variables
        indices_reordered = Vector{Int}()
        data_reordered = Vector{V}()
    end

    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    if n_p > 0
        # Call the recursive ManifoldBallTreeBalanced builder
        build_ManifoldBallTreeBalanced( mani, 1, data, data_reordered, hyper_spheres, metric, 
                                        indices, indices_reordered, 1,  length(data), 
                                        tree_data, array_buffs, reorder )
    end

    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    ManifoldBallTreeBalanced(mani, storedata ? data : similar(data, 0), hyper_spheres, indices, metric, tree_data, reorder)
end

function ManifoldBallTreeBalanced(  mani::AbstractManifold,
                            data::AbstractVector{T},
                            metric::M = Euclidean();
                            leafsize::Int = 10,
                            storedata::Bool = true,
                            reorder::Bool = true,
                            reorderbuffer::Matrix{T} = Matrix{T}(undef, 0, 0)) where {T <: AbstractFloat, M <: DST.Metric}
    #
    dim = size(data, 1)
    # npoints = size(data, 2)
    points = copy_svec(T, data, Val(dim))
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{dim,T}}()
    else
        reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    ManifoldBallTreeBalanced(mani, points, metric, leafsize = leafsize, storedata = storedata, reorder = reorder,
            reorderbuffer = reorderbuffer_points)
end

# Recursive function to build the tree.
function build_ManifoldBallTreeBalanced(
        mani::AbstractManifold,
        index::Int,
        data::AbstractVector{V},
        data_reordered::Vector{V},
        hyper_spheres::AbstractVector{<:ManifoldHyperSphere{N,T}},
        metric::Metric,
        indices::AbstractVector{Int},
        indices_reordered::AbstractVector{Int},
        low::Int,
        high::Int,
        tree_data::TreeData,
        array_buffs::ArrayPartitionBuffers{N,T},
        reorder::Bool ) where {V <: AbstractVector, N, T}
    #
    n_points = high - low + 1 # Points left
    if n_points <= tree_data.leafsize
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        tree_data.lleaf[index] = low
        tree_data.hleaf[index] = high
        tree_data.lchild[index] = low
        tree_data.rchild[index] = 0 # perhaps `= high`
        # Create bounding sphere of points in leaf node by brute force
        hyper_spheres[index] = create_bsphere(mani, data, metric, indices, low, high, array_buffs)
        return
    end

    # Find split such that one of the sub trees has 2^p points
    # and the left sub tree has more points
    mid_idx = find_split_balanced(high, low)

    # Brute force to find the dimension with the largest spread
    split_dim = find_largest_spread(data, indices, low, high)

    # Sort the data at the mid_idx boundary using the split_dim
    # to compare
    select_spec!(indices, mid_idx, low, high, data, split_dim)

    # if the left sub-tree is just one leaf, don't make a new non-leaf
    # node for it, just point left_idx directly to the leaf itself.
    left  = mid_idx   <= low  ? low  : (tree_data.next[]+=1; tree_data.next[]-1)
    right = mid_idx+1 >= high ? high : (tree_data.next[]+=1; tree_data.next[]-1)
    tree_data.lleaf[index]  = low
    tree_data.hleaf[index]  = high
    tree_data.lchild[index] = left
    tree_data.rchild[index] = right

    build_ManifoldBallTreeBalanced(mani, getleft(index), data, data_reordered, hyper_spheres, metric,
                    indices, indices_reordered, low, mid_idx,
                    tree_data, array_buffs, reorder)

    build_ManifoldBallTreeBalanced(mani, getright(index), data, data_reordered, hyper_spheres, metric,
                    indices, indices_reordered, mid_idx+1, high,
                    tree_data, array_buffs, reorder)

    # Finally create bounding hyper sphere from the two children's hyper spheres
    hyper_spheres[index]  =  create_bsphere(mani, metric, hyper_spheres[getleft(index)],
                                            hyper_spheres[getright(index)],
                                            array_buffs)
end

# function _knn(tree::ManifoldBallTreeBalanced,
#               point::AbstractVector,
#               best_idxs::AbstractVector{Int},
#               best_dists::AbstractVector,
#               skip::F) where {F}
#     knn_kernel!(tree, 1, point, best_idxs, best_dists, skip)
#     return
# end


# function knn_kernel!(tree::ManifoldBallTreeBalanced{V},
#                            index::Int,
#                            point::AbstractArray,
#                            best_idxs::AbstractVector{Int},
#                            best_dists::AbstractVector,
#                            skip::F) where {V, F}
#     if isleaf(tree.tree_data.n_internal_nodes, index)
#         add_points_knn!(best_dists, best_idxs, tree, index, point, true, skip)
#         return
#     end

#     left_sphere = tree.hyper_spheres[getleft(index)]
#     right_sphere = tree.hyper_spheres[getright(index)]

#     left_dist = max(zero(eltype(V)), evaluate(tree.metric, point, left_sphere.center) - left_sphere.r)
#     right_dist = max(zero(eltype(V)), evaluate(tree.metric, point, right_sphere.center) - right_sphere.r)

#     if left_dist <= best_dists[1] || right_dist <= best_dists[1]
#         if left_dist < right_dist
#             knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, skip)
#             if right_dist <=  best_dists[1]
#                  knn_kernel!(tree, getright(index), point, best_idxs, best_dists, skip)
#              end
#         else
#             knn_kernel!(tree, getright(index), point, best_idxs, best_dists, skip)
#             if left_dist <=  best_dists[1]
#                  knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, skip)
#             end
#         end
#     end
#     return
# end

# function _inrange(tree::ManifoldBallTreeBalanced{V},
#                   point::AbstractVector,
#                   radius::Number,
#                   idx_in_ball::Union{Nothing, Vector{Int}}) where {V}
#     ball = ManifoldHyperSphere(convert(V, point), convert(eltype(V), radius))  # The "query ball"
#     return inrange_kernel!(tree, 1, point, ball, idx_in_ball)  # Call the recursive range finder
# end

# function inrange_kernel!(tree::ManifoldBallTreeBalanced,
#                          index::Int,
#                          point::AbstractVector,
#                          query_ball::ManifoldHyperSphere,
#                          idx_in_ball::Union{Nothing, Vector{Int}})

#     if index > length(tree.hyper_spheres)
#         return 0
#     end

#     sphere = tree.hyper_spheres[index]

#     # If the query ball in the bounding sphere for the current sub tree
#     # do not intersect we can disrecard the whole subtree
#     if !intersects(tree.metric, sphere, query_ball)
#         return 0
#     end

#     # At a leaf node, check all points in the leaf node
#     if isleaf(tree.tree_data.n_internal_nodes, index)
#         return add_points_inrange!(idx_in_ball, tree, index, point, query_ball.r, true)
#     end

#     count = 0

#     # The query ball encloses the sub tree bounding sphere. Add all points in the
#     # sub tree without checking the distance function.
#     if encloses(tree.metric, sphere, query_ball)
#         count += addall(tree, index, idx_in_ball)
#     else
#         # Recursively call the left and right sub tree.
#         count += inrange_kernel!(tree,  getleft(index), point, query_ball, idx_in_ball)
#         count += inrange_kernel!(tree, getright(index), point, query_ball, idx_in_ball)
#     end
#     return count
# end


# test case
# pts = [
#     1.6428412203258511
#     -0.4823265406855113
#      0.4354221188230193
#      1.908228908562008
#      0.9791603637197599
#      1.0798652993450037
#      1.0875113872287496
#      1.2019334066681153
#      0.013282018302654335
#      0.965302261228411   
# ]