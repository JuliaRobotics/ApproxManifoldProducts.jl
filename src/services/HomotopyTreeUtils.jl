

# _getleft(i::Integer, N) = 2*i + (2*i < N ? 0 : 1)
# _getright(i::Integer, N) = _getleft(i,N) + 1

# either tree or leaf kernel, if larger than N
leftIndex(mt::HomotopyDensity, krnIdx::Int = 1) = childIndices(mt, krnIdx).left
    # return 2 * krnIdx + (2 * krnIdx < length(mt) ? 0 : 1)

rightIndex(mt::HomotopyDensity, krnIdx::Int) = childIndices(mt, krnIdx).right
    #leftIndex(mt, krnIdx) + 1

# EXPERIMENTAL, untested, likely buggy
function childIndices(
    mt::HomotopyDensity, 
    krnIdx::Int;
    mixturedepth::Int = 999,
)
    N = Npts(mt)
    btleft = 2 * krnIdx
    # e.g. for N=length(data)=32, left child of 1*2 = 2, and left child of 2*2=4, whose left child is 4*2 = 8, similarly 8*2=16.  
    #  Now the left child of node 16*2 = 32, which is the first leaf node (but careful with index == N)
    #  i.e. right child of node 15 is 2*15+1 = 31, so 15's right child (31) is the last nonleaf
    isleaf = N <= btleft
    # Before BeliefTreeIndices nonisleaf are [1..N], while isleaf are [N+1..2N].
    left = btleft + (isleaf ? 1 : 0) 
    nonleaf_left = isleaf ? -1 : btleft
    leaf_left = isleaf ? nonleaf_left : -1
    right = left + 1 
    nonleaf_right = isleaf ? -1 : nonleaf_left + 1
    leaf_right = isleaf ? nonleaf_left + 1 : -1
    # return a pseudo type representing a composite index of the belief tree
    left_ci = (;
        nonleaf_left,
        leaf_left,
        isleaf,
        # TBD permuted indices?
    )
    right_ci = (;
        nonleaf_right,
        leaf_right,
        isleaf,
        # TBD permuted indices?
    )
    return (;
        left_ci,
        right_ci,
        # legacy values below
        N,
        left,
        right, 
    )
end




# check for existence in tree or leaves
function exists_BTLabel(hode::HomotopyDensity, idx::Int)
    N = Npts(hode)
    if idx < N
        return isassigned(hode.tree_kernels, idx)
    else
        return isassigned(hode.leaf_kernels, idx - N + 1)
    end
end

function isLeaf_BTLabel(mt::HomotopyDensity, idx::Int)
    if exists_BTLabel(mt, leftIndex(mt, idx))
        return false
    elseif exists_BTLabel(mt, rightIndex(mt, idx))
        # TODO likely not needed to check for right child existence
        return false
    else
        return true
    end
end



"""
    $SIGNATURES

Default returns leaf kernel associated with permuted input data element `i` (i.e. `permuted=true`).
but returns the leaf_kernel inverse permuted `i` when `permuted=false` (i.e. similar to unsorted input data).

DevNotes:
- Very bad practice to have duplicate of .data[.permuted] deepcopied into .leaf_kernels
  - Makes unpermuted lookup really slow among the torrent of other issues.  FIXME
"""
function getKernelLeaf(
    mt::HomotopyDensity, 
    i::Int, 
    permuted::Bool = true
)
    invpermute(s::Int) = findfirst(==(s), mt.permute)
    if permuted
        return mt.leaf_kernels[i]
    else
        return mt.leaf_kernels[invpermute(i)]
    end
end

"""
    $SIGNATURES

Return leaf kernels as tree kernel types, using regular `[1..N]` indexing].

Notes:
- use `permute=true` (default) for sorted index retrieval.
"""
getKernelLeafAsTreeKer(
    mtr::HomotopyDensity{L, M, P, HL, HT},
    idx::Int,
    permuted::Bool = false,
) where {M, L, P, HL, HT} = convert(HT, getKernelLeaf(mtr, (idx - 1) % Npts(mtr) + 1, permuted))

"""
    $SIGNATURES

Return kernel from tree by binary tree index, and convert leaf kernels to tree kernel types if necessary.

Notes:
- BinaryTree (BT) index goes from root=1 to largest leaf 2*N

See also: [`getKernelLeafAsTreeKer`](@ref)
"""
function getKernelTree(
    hode::HomotopyDensity{L, M, P, HL, HT},
    currIdx::Int,
    # must return sorted given name signature "Tree"
    permuted::Bool = false,
    cov_continuation::Bool = false,
) where {M, L, P, HL, HT}
    #
    N = Npts(hode)
    # BinaryTree (BT) index goes from root=1 to largest leaf 2*N
    if currIdx < N
        # cov_continuation correction so that we may build trees with sensible convariance to bandwidth transition from root to leaf
        raw_ker = hode.tree_kernels[currIdx]
        if cov_continuation
            # depth of this index
            ances_depth = floor(Int, log2(currIdx))
            # how many leaf offsp
            offsp_depth = log2(length(hode.segments[currIdx]))
            # get approx continuous depth fraction of this index
            λ = (ances_depth) / (ances_depth + offsp_depth)
            # mean bandwidth of all leaf children
            leafIdxs = hode.segments[currIdx] .|> s -> findfirst(==(s), hode.permute)
            leafIdxs .+= N
            # TBD, why permuted hard false here, maybe because tree nodes not leaves?
            bws = [cov(getKernelTree(hode, lidx, false)) for lidx in leafIdxs] 
            # FIXME is a parallel transport needed between different kernel covariances that each exist in different tangent spaces
            mean_bw = Matrix(mean(bws)) # FIXME upgrade to on-manifold mean
            # corrected cov varies from root (only Monte Carlo cov est) to leaves (only selected bandwdith)
            nC = (1 - λ) * (cov(raw_ker)) + λ * mean_bw
            # return a new kernel with cov_continuation, of tree kernel type
            # FIXME, remember partial information
            kernelType = getfield(ApproxManifoldProducts, HT.name.name)
            partial = _getprl(raw_ker)
            μ = mean(raw_ker)
            M_, reprl, partl_cb = getManifoldPartial(getManifold(hode), _tuple(partial), μ)
            kernelType(μ, nC, hode.weights[currIdx]; partial, partl_cb)
        else
            raw_ker
        end
    else
        getKernelLeafAsTreeKer(hode, currIdx, permuted)
    end
end

## =================================================================================
## Tree build functions
## =================================================================================


"""
    $SIGNATURES

Notes:
- Bandwidths for leaves (i.e. `kernel_bw`) must be passed in as covariances when `ConcentratedGaussianKernel`.

DevNotes:
- Design Decision 24Q1, Manellic.MvNormalKernel bandwidth defs should ALWAYS ONLY BE covariances, because
  - Vision state is multiple bandwidth kernels including off diagonals in both tree or leaf kernels
  - Hybrid parametric to leafs covariance continuity
  - https://github.com/JuliaStats/Distributions.jl/blob/a9b0e3c99c8dda367f69b2dbbdfa4530c810e3d7/src/multivariate/mvnormal.jl#L220-L224
"""
function buildTree_Manellic!(
    M::AbstractManifold,
    r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
    N = length(r_PP),
    weights::AbstractVector{<:Real} = ones(N) .* (1 / N),
    kernel = ConcentratedGaussianKernel,
    kernel_bw = nothing, # TODO
    partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {P <: AbstractArray}
    #
    D = manifold_dimension(M)
    CV = SMatrix{D, D, Float64, D * D}(diagm(ones(D)))
    prlcb = if isnothing(partl_cb) && !isnothing(partial)
        M_, reprl, cb = getManifoldPartial(M, partial)
        cb
    else
        partl_cb
    end
    tknlT = kernel(r_PP[1], CV; partial=_tuple(partial), partl_cb=prlcb) |> typeof

    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)
    _legacybw(::Nothing) = CV

    lCV = _legacybw(kernel_bw)

    lknlT = kernel(r_PP[1], lCV; partial = _tuple(partial), partl_cb=prlcb) |> typeof

    # kernel scale

    # leaf kernels
    lkern = Vector{lknlT}(undef, N)
    for i = 1:N
        nkr = kernel(r_PP[i], lCV; partial = _tuple(partial), partl_cb=prlcb)
        lkern[i] = nkr
    end

    _hode = ApproxManifoldProducts.HomotopyDensity{
        _tuple(partial),
    }(;
        manifold = M,
        data = r_PP,
        weights,
        leaf_kernels = lkern,                      # leaf_kernels
        tree_kernels = Vector{tknlT}(undef, N),    # tree_kernels
    );

    #
    tosort_leaves = buildTree_Manellic!(
        _hode,
        1, # start at root
        1, # spanning all data
        N; # to end of data
        kernel,
        kernel_bw,
        partial = _tuple(partial),
        partl_cb = prlcb,
    )

    # manual reset leaves in the order discovered
    permute!(tosort_leaves.leaf_kernels, tosort_leaves.permute)

    return tosort_leaves
end

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
    # HACK returns if keepbranching==false, also mid=-1, knl=nothing
    mid_idx, keepbranching = splitsortBinary!(
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
    if !keepbranching
        return hode
    end

    # # set tree kernel
    # # FIXME, THIS USED TO BE BELOW recursive subtree build
    # tkT = eltype(hode.tree_kernels)
    # hode.tree_kernels[index] = tkT(knl; partl_cb)
    # hode.segments[index] = sido # Set(ido)     

    
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


    keepbranching = true

    # take a slice of data
    npts = high - low + 1
    idc = low:high
    
    # according to current index permutation (i.e. sort data as you build the tree)
    ido = view(hode.permute, idc)
    
    # secondary recursion termination case, seems odd to have two terminations 
    # FIXME, this will likely fail if leaf_size is not 1, since mid_idx will be wrong
    if npts <= leaf_size
        keepbranching = false
        # HACK mid=-1, knl=nothing if keepbranching false 
        return -1, keepbranching
    end


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


    # primary recursion termination case
    #  occurs after permute sort modifications in recursion stack 
    #  this prevents an overshoot in index...
    if (Npts(hode) <= index)
        keepbranching = false
    else
        # set tree kernel
        # NOTE, THIS USED TO BE BELOW recursive subtree build
        tkT = eltype(hode.tree_kernels)
        hode.tree_kernels[index] = tkT(knl; partl_cb)
        hode.segments[index] = Set(ido)     
    end

    return mid_idx, keepbranching
end
