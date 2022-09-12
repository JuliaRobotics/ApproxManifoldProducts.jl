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




# A ManifoldBalancedBallTree (also called Metric tree) is a tree that is created
# from successively splitting points into surrounding hyper spheres
# which radius are determined from the given metric.
# The tree uses the triangle inequality to prune the search space
# when finding the neighbors to a point,
struct ManifoldBalancedBallTree{V <: AbstractArray, T, M <: DST.Metric} # <: NNTree{V,M}
    """ tree points exist on some manifold `<:Manifolds.AbstractManifold` """
    manifold::T
    """ data points associated with this tree """
    data::Vector{V}
    """ Each hyper sphere bounds its children """
    hyper_spheres::Vector{<:ManifoldHyperSphere} 
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
struct ArrayPartitionBuffers{N,T <: Union{<:AbstractArray, <:Real}}
    center::T
end





# Recursive function to build the tree.
function build_ManifoldBalancedBallTree(
        mani::AbstractManifold,
        index::Int,
        data::AbstractVector{V},
        data_reordered::AbstractVector{V},
        hyper_spheres::AbstractVector{<:ManifoldHyperSphere},
        metric::DST.Metric,
        indices::AbstractVector{Int},
        indices_reordered::AbstractVector{Int},
        low::Int,
        high::Int,
        tree_data::TreeDataBalanced,
        array_buffs::ArrayPartitionBuffers,
        reorder::Bool 
    ) where {V <: AbstractArray}
    #
    @info "@24R1" index length(tree_data.lchild) length(tree_data.rchild)
    n_points = high - low + 1 # Points left
    if n_points <= tree_data.leafsize
        if reorder
            NNR.reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        tree_data.lleaf[index] = low
        tree_data.hleaf[index] = high
        tree_data.lchild[index] = low
        tree_data.rchild[index] = 0 # perhaps `= high`
        # Create bounding sphere of points in leaf node by brute force
        @info "RIGHT" typeof(hyper_spheres) typeof(array_buffs)
        nsph = NNR.create_bsphere(mani, data, metric, indices, low, high, array_buffs)
        @info "OKAY" typeof(nsph) index isdefined(hyper_spheres, index)
        hyper_spheres[index] = nsph
        return
    end

    # Find split such that one of the sub trees has 2^p points
    # and the left sub tree has more points
    mid_idx = find_split_balanced(high, low)

    # Brute force to find the dimension with the largest spread
    split_dim = NNR.find_largest_spread(data, indices, low, high)

    # Sort the data at the mid_idx boundary using the split_dim
    # to compare
    NNR.select_spec!(indices, mid_idx, low, high, data, split_dim)

    # if the left sub-tree is just one leaf, don't make a new non-leaf
    # node for it, just point left_idx directly to the leaf itself.
    left  = mid_idx   <= low  ? low  : (tree_data.next[]+=1; tree_data.next[]-1)
    right = mid_idx+1 >= high ? high : (tree_data.next[]+=1; tree_data.next[]-1)
    tree_data.lleaf[index]  = low
    tree_data.hleaf[index]  = high
    tree_data.lchild[index] = left
    tree_data.rchild[index] = right

    build_ManifoldBalancedBallTree(mani, NNR.getleft(index), data, data_reordered, hyper_spheres, metric,
                    indices, indices_reordered, low, mid_idx,
                    tree_data, array_buffs, reorder)

    build_ManifoldBalancedBallTree(mani, NNR.getright(tree_data, index), data, data_reordered, hyper_spheres, metric,
                    indices, indices_reordered, mid_idx+1, high,
                    tree_data, array_buffs, reorder)

    # Finally create bounding hyper sphere from the two children's hyper spheres
    hyper_spheres[index]  =  NNR.create_bsphere(mani, metric, hyper_spheres[NNR.getleft(tree_data, index)],
                                            hyper_spheres[NNR.getright(tree_data, index)],
                                            array_buffs)
end


"""
    ManifoldBalancedBallTree(data [, metric = Euclidean(); leafsize = 10, reorder = true]) -> ManifoldBalancedballtree

Creates a `ManifoldBalancedBallTree` from the data using the given `metric` and `leafsize`.
"""
function ManifoldBalancedBallTree(
                    mani::AbstractManifold,
                    data::AbstractVector{V},
                    metric::M = DST.Euclidean();
                    leafsize::Int = 10,
                    reorder::Bool = true,
                    storedata::Bool = true,
                    reorderbuffer::AbstractVector{V} = Vector{V}()) where {V <: AbstractArray, M <: DST.Metric}
    #
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    _getMSimilar(s::AbstractVector) = MVector{length(V), eltype(V)}(undef)
    _getMSimilar(s::AbstractMatrix) = MMatrix{size(s,1), size(s,2), eltype(V)}(undef)
    _getMSimilar(s::ArrayPartition) = ArrayPartition(map(x->_getMSimilar(x),s.x)...)

    tree_data = TreeDataBalanced(data, leafsize)
    n_d = length(V)
    n_p = length(data)

    # dTy = NNR.get_T(eltype(V))
    mvc = _getMSimilar(data[1])
    array_buffs = ArrayPartitionBuffers{length(V), typeof(mvc)}(mvc)
    indices = collect(1:n_p)

    # Bottom up creation of hyper spheres so need spheres even for leafs)
    # @info "HYPS" typeof(data[1]) typeof(mvc)
    hyper_spheres = Vector{ManifoldHyperSphere{typeof(mvc),eltype(V)}}(undef, 2*n_p) # tree_data.n_internal_nodes + tree_data.n_leafs)

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

    if metric isa DST.UnionMetrics
        p = DST.parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    if n_p > 0
        # Call the recursive ManifoldBalancedBallTree builder
        @info "TYPES" typeof(data) typeof(data_reordered) typeof(hyper_spheres)
        build_ManifoldBalancedBallTree( mani, 1, data, data_reordered, hyper_spheres, metric, 
                                        indices, indices_reordered, 1,  length(data), 
                                        tree_data, array_buffs, reorder )
    end

    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    ManifoldBalancedBallTree(mani, storedata ? data : similar(data, 0), hyper_spheres, indices, metric, tree_data, reorder)
end


function ManifoldBalancedBallTree(  mani::AbstractManifold,
                                    data::AbstractVector{T},
                                    w...;
                                    reorderbuffer::AbstractVector{T} = Vector{T}(undef, 0),
                                    kw...) where {T <: Number}
    #
    pts = Vector{SVector{1,T}}(undef, length(data))
    for (i,de) in enumerate(data)
        pts[i] = SA[de;]
    end
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{1,T}}()
    else
        error("reorderbuffer options not fully implemented yet")
        # reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    ManifoldBalancedBallTree(mani, pts, w...; reorderbuffer=reorderbuffer_points, kw...)
end


# function _knn(tree::ManifoldBalancedBallTree,
#               point::AbstractVector,
#               best_idxs::AbstractVector{Int},
#               best_dists::AbstractVector,
#               skip::F) where {F}
#     knn_kernel!(tree, 1, point, best_idxs, best_dists, skip)
#     return
# end


# function knn_kernel!(tree::ManifoldBalancedBallTree{V},
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

# function _inrange(tree::ManifoldBalancedBallTree{V},
#                   point::AbstractVector,
#                   radius::Number,
#                   idx_in_ball::Union{Nothing, Vector{Int}}) where {V}
#     ball = ManifoldHyperSphere(convert(V, point), convert(eltype(V), radius))  # The "query ball"
#     return inrange_kernel!(tree, 1, point, ball, idx_in_ball)  # Call the recursive range finder
# end

# function inrange_kernel!(tree::ManifoldBalancedBallTree,
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


function Base.show(io::IO, tree::ManifoldBalancedBallTree{V,M}) where {V,M}
    println(io, typeof(tree).name.name, "{")
    println(io, "    V = ", V)
    println(io, "    M = ", M)
    println(io, "  }")
    println(io, "  Number of points: ", length(tree.data))
    println(io, "  Dimensions:       ", manifold_dimension(tree.manifold))
    println(io, "  Metric:           ", tree.metric)
    print(io,   "  Reordered:        ", tree.reordered)
end

