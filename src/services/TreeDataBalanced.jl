
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

Base.@kwdef struct TreeDataBalanced
    """ Number of points in the last node """
    last_node_size::Int     = 0
    """ Number of points in each leaf node (except last) """
    leafsize::Int           = 0
    """ Number of leafs """
    n_leafs::Int            = 0
    """ Number of non leaf nodes """
    n_internal_nodes::Int   = 0
    cross_node::Int         = 0
    offset::Int             = 0
    offset_cross::Int       = 0
    last_full_node::Int     = 0
    lleaf::Vector{Int}      = Int[]
    hleaf::Vector{Int}      = Int[]
    lchild::Vector{Int}     = Int[]
    rchild::Vector{Int}     = Int[]
    pselect::Vector{Int}    = Int[]
    next::Base.RefValue{Int}= Ref(0)
end


function TreeDataBalanced(data::AbstractVector{V}, leafsize) where V
    @info "HERE" V typeof(data)
    n_dim, n_p = length(V), length(data)

    # If number of points is zero
    n_p === 0 && return TreeDataBalanced()

    n_leafs =  ceil(Integer, n_p / leafsize)
    n_internal_nodes = n_leafs - 1
    leafrow = floor(Integer, log2(n_leafs))
    cross_node = 2^(leafrow + 1)
    last_node_size = n_p % leafsize
    if last_node_size === 0
        last_node_size = leafsize
    end

    # This only happens when n_p / leafsize is a power of 2?
    if cross_node >= n_internal_nodes + n_leafs
        cross_node = div(cross_node, 2)
    end

    offset = 2(n_leafs - 2^leafrow) - 1
    k1 = (offset - n_internal_nodes - 1) * leafsize + last_node_size + 1
    k2 = -cross_node * leafsize + 1
    last_full_node = n_leafs + n_internal_nodes

    vit = zeros(Int, 2*length(data))
    vit[n_p+1:2n_p] = 1:n_p
    vir = similar(vit)
    vir[n_p+1:2n_p] = -1*ones(Int, n_p)

    TreeDataBalanced(;last_node_size, leafsize, n_leafs,
        n_internal_nodes, cross_node, offset=k1, offset_cross=k2, last_full_node,
        lleaf=vit, hleaf=vit, lchild=vit, rchild=vit )
end
