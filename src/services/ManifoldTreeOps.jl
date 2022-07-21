


# Split the tree such that one of the sub trees have +-1 an equal number of points
function find_split_balanced(high, low)
    (floor(Int,(low + high) / 2))
end

# Gets number of points in a leaf node, this is equal to leafsize for every node
# except the last node.
@inline function NNR.n_ps(idx::Int, td::TreeDataBalanced)
    if idx != td.last_full_node
        return td.leafsize
    else
        return td.last_node_size
    end
end

# Returns the index for the first point for a given leaf node.
@inline function NNR.point_index(idx::Int, td::TreeDataBalanced)
    if idx >= td.cross_node
        return td.offset_cross + idx * td.leafsize
    else
        return td.offset + idx * td.leafsize
    end
end

# Returns a range over the points in a leaf node with a given index
@inline function NNR.get_leaf_range(td::TreeDataBalanced, index)
    p_index = NNR.point_index(index, td)
    n_p =  NNR.n_ps(index, td)
    return p_index:p_index + n_p - 1
end

# Store all the points in a leaf node continuously in memory in data_reordered to improve cache locality.
# Also stores the mapping to get the index into the original data from the reordered data.
function NNR.reorder_data!(data_reordered::Vector{V}, data::AbstractVector{V}, index::Int,
                            indices::Vector{Int}, indices_reordered::Vector{Int}, tree_data::TreeDataBalanced) where {V}
    #
    for i in NNR.get_leaf_range(tree_data, index)
        idx = indices[i]
        data_reordered[i] = data[idx]
        # Saves the inverse n
        indices_reordered[i] = idx
    end
end


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

#