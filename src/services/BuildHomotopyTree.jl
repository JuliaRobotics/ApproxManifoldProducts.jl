
## =================================================================================
## Tree build functions
## =================================================================================



function buildTree_Manellic!(
    M::AbstractManifold,
    r_ker::AbstractVector{KL}; # vector of points referenced to the r_frame
    N = length(r_ker),
    weights::AbstractVector{<:Real} = ones(N) .* (1 / N),
    kernel = KL,
    kernel_bw = nothing, # TODO
    # partial = ??? TBD -- it should already be in the kernels
) where {KL <: ConcentratedGaussianKernel}
    #
    _μT() = typeof(mean(r_ker[1]))
    D = manifold_dimension(M)
    CV = SMatrix{D, D, Float64, D * D}(collect(cov(r_ker[1])))
    _KLT(k) = getfield(ApproxManifoldProducts, k.name.name)
    _KLT(k::UnionAll) = k
    KLT = _KLT(kernel)
    KT = KLT(mean(r_ker[1]), CV) |> typeof

    r_PP = Vector{_μT()}(undef, N)

    # leaf kernels
    lkern = Vector{KL}(undef, N)
    for i = 1:N
        r_PP[i] = mean(r_ker[i])
        lkern[i] = if isnothing(kernel_bw)
            r_ker[i]
        else
            updateKernelBW(r_ker[i], kernel_bw) # TODO handle vector of kernel_bws
        end
    end

    mtree = HomotopyDensity{
        _getprl(r_ker[1]),
    }(;
        manifold = M,
        data = r_PP,
        weights,
        leaf_kernels = lkern,
        tree_kernels = Vector{KT}(undef, N),
    )

    #
    tosort_leaves = buildTree_Manellic!(
        mtree,
        1, # start at root
        1, # spanning all data
        N; # to end of data
        kernel = KLT,
        kernel_bw,
    )

    # # manual reset leaves in the order discovered
    # permute!(tosort_leaves.leaf_kernels, tosort_leaves.geometric_permute[1])

    return tosort_leaves
end


function buildTree_Manellic!(
    hode::HomotopyDensity,
    index::Integer, # tree node root=1,left=2n+corr,right=left+1
    low::Integer,   # bottom index of segment
    high::Integer;  # top index of segment;
    kernel = nothing, # TODO obsolete
    kernel_bw = nothing,
    leaf_size = 1,
    partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,  # TODO remove, get from hode internally
    partl_cb::Union{Nothing, <:Function} = nothing,
)
    # DX bug stop, FIXME remove, ensure bugfree with tests -- see testManifoldTreeConstr.jl and others
    if index < 0
        @error "DX bug stop, details:" N leaf_size index low high partial
        error("HomotopyDensity tree build error, index=($index) should not be negative")
    end

    # Must always sort down into leaf_size pool (which might be > 1) before terminating recursion
    mid_idx, sml, big = splitsortBinary!(
        hode,
        low,
        high,
        index;
        leaf_size,
        kernel_bw,
        partial, # TODO remove, get from hode internally
        partl_cb,
    )

    # Terminate recursion as determined by splitsort, and after necessary sort in leaf nodes of tree
    if mid_idx < 0
        return hode
    end

    # recursively check need for left subtree
    lftidx = leftIndex(hode, index)
    hode.geometric_permute[lftidx] = collect(sml)
    if low < mid_idx
        # build of new child node requires expansion of geometric permute field
        buildTree_Manellic!(
            hode,
            lftidx,
            low,
            mid_idx;
            kernel_bw,
            leaf_size,
            partial,
            partl_cb,
        )
        sml = hode.geometric_permute[lftidx] # update sml since tree build will continue
    end
    # recursively check need for right subtree
    rhtidx = rightIndex(hode, index)
    hode.geometric_permute[rhtidx] = collect(big)
    if (mid_idx + 1) < high
        # build of new child node requires expansion of geometric permute field
        buildTree_Manellic!(
            hode,
            rhtidx,
            mid_idx + 1,
            high;
            kernel_bw,
            leaf_size,
            partial,
            partl_cb,
        )
        big = hode.geometric_permute[rhtidx] # update big since tree build will continue
    end

    # at start, .geometric_permute[1] is spread over all data 1:N and populated by HomotopyDensity constructor
    # vcat ensures sml, big are buffered for in-place "swap" of the slice portion of geometric_permute, else elements overwritten prematurely
    # TODO, not sure if this step is needed or if this is the right place for this call
    # must set once for each node, above each child node was permuted and now the parent list must also be permuted so that sorting propagates to top list
    hode.geometric_permute[index] .= vcat(sml, big)

    return hode
end


## ================================================================================
## Binary tree split and sort
## ================================================================================



function splitsortBinary!(
    hode::HomotopyDensity,
    low::Integer,
    high::Integer,
    index::Integer;
    leaf_size::Integer = 1,
    kernel_bw = nothing,
    partial::Union{Nothing, <:Tuple},
    partl_cb,
)

    N = Npts(hode)
    # take a slice of data
    idc = low:high
    
    # according to current index permutation (i.e. sort data as you build the tree)
    gido = hode.geometric_permute[index]
    # gido = view(hode.geometric_permute[index], 1:length(idc)) #idc)

    # split the slice of order-permuted data
    _, mask, midoffset, p, bw = splitPointsEigen(
        getManifold(hode),
        view(hode.data, gido);
        kernel_bw,
        partial,
    )
    imask = xor.(mask, true)
    
    # sort the data as 'small' and 'big' elements either side of the eigen split
    # towards accending (in-place) reorder of the slice portion
    # in-place replacement requires a temporary buffer -- achieved by vcat, else can use collect here
    big = view(gido, mask)  |> collect
    sml = view(gido, imask) |> collect
    # TODO, reduce mem with gido[1:nsml] .= sml ... instead :::: vcat buffers elements for in-place "swap", else elements overwritten prematurely 
    gido .= vcat(sml, big)  

    # store tree kernel and segment indices; after sorting
    if (index <= N)
        # set tree kernel
        # NOTE, THIS USED TO BE AFTER recursive subtree build
        tkT = eltype(hode.tree_kernels)
        knl = ConcentratedGaussianKernel(
            p, bw, sum(view(hode.weights, gido)); 
            partial, partl_cb
        )
        hode.tree_kernels[index] = tkT(knl; partl_cb)
    end
    
    # recursion termination case
    # geometric split instead of data split (must happen in cosort classification labeling) 
    # TBD, untested leaf_size is not 1
    npts = high - low + 1
    mid_idx = if (npts <= leaf_size) || (N < index)
        -1
    else
        # TODO update gido since tree build will continue
        # return valid mid_idx
        low + midoffset
    end

    return mid_idx, sml, big
end
