

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
    # FIXME, isleaf for right children only (can happen when doing geometric split)
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
    # right unbalanced trees are also possible, so cannot base false on lack of left
    if exists_BTLabel(mt, leftIndex(mt, idx)) || exists_BTLabel(mt, rightIndex(mt, idx))
        return false
    end
    return true
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


##