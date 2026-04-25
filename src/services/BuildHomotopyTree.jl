
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
        permute = collect(1:N),
        leaf_kernels = lkern,
        tree_kernels = Vector{KT}(undef, N),
        segments = Vector{Set{Int}}(undef, N),
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

    # manual reset leaves in the order discovered
    permute!(tosort_leaves.leaf_kernels, tosort_leaves.permute)

    return tosort_leaves
end


function buildTree_Manellic!(
    hode::HomotopyDensity,
    index::Integer, # tree node root=1,left=2n+corr,right=left+1
    low::Integer,   # bottom index of segment
    high::Integer;  # top index of segment;
    kernel = MvNormal,
    kernel_bw = nothing,
    leaf_size = 1,
    partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,  # TODO remove, get from hode internally
    partl_cb::Union{Nothing, <:Function} = nothing,
)
    # DX bug stop, FIXME remove, ensure bugfree with tests
    if index < 0
        @error "details:" N leaf_size index low high partial
        error("HomotopyDensity tree build error, index=($index) should not be negative")
    end

    # Must always sort down into leaf_size pool (which might be > 1) before terminating recursion
    mid_idx, stopbranching = splitsortBinary!(
        hode,
        low,
        high,
        index;
        leaf_size,
        kernel,
        kernel_bw,
        partial, # TODO remove, get from hode internally
        partl_cb,
    )

    # Terminate recursion as determined by splitsort, and after necessary sort in leaf nodes of tree
    if stopbranching
        return hode
    end
    # !(Npts(hode) <= index) && error("DX bug stop. This should not have happened since npts<=leaf_size should have stopped tree build recursion, but now Npts(hode)=$(Npts(hode)) > index=$(index)")


    # recursively check need for left subtree
    if low < mid_idx
        buildTree_Manellic!(
            hode,
            leftIndex(hode, index),
            low,
            mid_idx;
            kernel,
            kernel_bw,
            leaf_size,
            partial,
            partl_cb,
        )
    end
    # recursively check need for right subtree
    if (mid_idx + 1) < high
        buildTree_Manellic!(
            hode,
            rightIndex(hode, index),
            mid_idx + 1,
            high;
            kernel,
            kernel_bw,
            leaf_size,
            partial,
            partl_cb,
        )
    end

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
    kernel,
    kernel_bw = nothing,
    partial::Union{Nothing, <:Tuple},
    partl_cb,
)

    _legacybw(s::Nothing) = s
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)

    # take a slice of data
    idc = low:high
    
    # according to current index permutation (i.e. sort data as you build the tree)
    ido = view(hode.permute, idc)

    # split the slice of order-permuted data
    _, mask, knl = splitPointsEigen(
        getManifold(hode),
        view(hode.data, ido),
        view(hode.weights, ido);
        kernel,
        kernel_bw = _legacybw(kernel_bw),
        partial,
        partl_cb,
    )
    imask = xor.(mask, true)

    # sort the data as 'small' and 'big' elements either side of the eigen split
    big = view(ido, mask)  |> collect
    sml = view(ido, imask) |> collect
    # inplace reorder the slice portion of hode.permute towards accending
    _ido = SA[sml...; big...]
    # ido .= SA[sml...; big...]
    for (i,v) in enumerate(_ido)
        ido[i] = v
    end
    mid_idx = low + sum(imask) - 1

    # store tree kernel and segment indices; after sorting
    if (index <= Npts(hode))
        # set tree kernel
        # NOTE, THIS USED TO BE AFTER recursive subtree build
        tkT = eltype(hode.tree_kernels)
        hode.tree_kernels[index] = tkT(knl; partl_cb)
        hode.segments[index] = Set(ido)
    end

    # recursion termination case
    # TBD, untested leaf_size is not 1
    npts = high - low + 1
    stopbranching = (npts <= leaf_size) || (Npts(hode) < index)

    return mid_idx, stopbranching
end
