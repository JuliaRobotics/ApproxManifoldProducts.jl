

# either tree or leaf kernel, if larger than N
leftIndex(::HomotopyDensity, idx::Int = 1) = 2 * idx
rightIndex(hode::HomotopyDensity, idx::Int) = leftIndex(hode, idx) + 1


# check for existence in tree or leaves
function isassigned(hode::HomotopyDensity, idx::Int)
    if idx < 2*(Npts(hode)+1)
        return Base.isstored(hode.structure, idx)
    else
        return false
    end
end

# right unbalanced trees are also possible, so cannot base false on lack of left
isLeaf_BTLabel(hode::HomotopyDensity, idx::Int) = !isassigned(hode, leftIndex(hode, idx)) || !isassigned(hode, rightIndex(hode, idx))



"""
    $SIGNATURES

Default returns leaf kernel associated with permuted input data element `i` (i.e. `permuted=true`).
but returns the leaf_kernel inverse permuted `i` when `permuted=false` (i.e. similar to unsorted input data).

DevNotes:
- FIXME Very bad practice to have duplicate of .points[.structure[1]] deepcopied into .leaf_kernels
  - Makes unpermuted lookup really slow among the torrent of other issues.
"""
function getKernelLeaf(
    hode::HomotopyDensity, 
    i::Int, 
    permuted::Bool = true
)
 
    idx = if permuted
        # FIXME, can only use structure[1] when leaf_size=1
        hode.structure[1][i]
    else
        i
    end

    # FIXME refactor in transit, use HR{,,,A} to extract leaf and minor relation
    # TODO consolidate with uniBW()
    cv_ = if 1 == SparseArrays.nnz(hode.minors_detail)
        hode.minors_detail[1]
    else
        hode.minors_detail[i]
    end

    hr = hode.reprkind
    partial = getPartial(hr)
    mn = hode.points[idx] # mean(lv) # 

    # FIXME, hack before reworking partials to common trait -- 
    #  partial and partl_cb elsewhere assumed to travel together, not recreated post-hoc
    _, _, partl_cb = getManifoldPartial(getManifold(hode), _tuple(partial), mn)
    return getReprType(hr)(mn, cv_; partial, partl_cb)
end

"""
    $SIGNATURES

Return leaf kernels as tree kernel types, using regular `[1..N]` indexing].

Notes:
- use `permute=true` (default) for sorted index retrieval.
"""
function getKernelLeafAsTreeKer(
    mtr::HomotopyDensity{H, P},
    idx::Int,
    permuted::Bool = false,
) where {H, P}
    reprT = getReprType(mtr.reprkind)
    partial = getPartial(mtr.reprkind)
    mani = getManifold(mtr)
    lidx = (idx - 1) % Npts(mtr) + 1
    lidx_ = if permuted
        # FIXME, can only use structure[1] when leaf_size=1
        mtr.structure[1][lidx]
    else
        lidx
    end
    lk = getKernelLeaf(mtr, lidx_, permuted)
    μ = mean(lk)
    manil_, _, partl_cb = getManifoldPartial(mani, partial, μ)
    reprT(μ, cov(lk), mtr.weights[lidx_]; partial, partl_cb)
end

"""
    $SIGNATURES

Return kernel from tree by binary tree index, and convert leaf kernels to tree kernel types if necessary.

Notes:
- BinaryTree (BT) index goes from root=1 to largest leaf 2*N

See also: [`getKernelLeafAsTreeKer`](@ref)
"""
function getKernelTree(
    hode::HomotopyDensity{H, P},
    currIdx::Int,
    # must return sorted given name signature "Tree"
    permuted::Bool = false,
    cov_continuation::Bool = false,
) where {H, P}
    #
    N = Npts(hode)
    partial = getPartial(hode)
    reprT = getReprType(hode.reprkind)
    # BinaryTree (BT) index goes from root=1 to largest leaf 2*N
    return if isassigned(hode, currIdx) && !isLeaf_BTLabel(hode, currIdx)
        # cov_continuation correction so that we may build trees with sensible convariance to bandwidth transition from root to leaf
        μ = hode.majors_element[currIdx]
            # FIXME, hack before reworking partials to common trait -- 
            #  partial and partl_cb elsewhere assumed to travel together, not recreated post-hoc
            _, _, partl_cb = getManifoldPartial(getManifold(hode), _tuple(partial), μ)
        raw_ker = reprT(
            μ, 
            hode.majors_detail[currIdx], 
            hode.majors_coeff[currIdx];
            partial, partl_cb
        )
        if cov_continuation
            # depth of this index
            ances_depth = floor(Int, log2(currIdx))
            # how many leaf offsp
            offsp_depth = log2(length(hode.structure[currIdx]))
            # get approx continuous depth fraction of this index
            λ = (ances_depth) / (ances_depth + offsp_depth)
            # mean bandwidth of all leaf children
            leafIdxs = hode.structure[currIdx] .|> s -> findfirst(==(s), hode.structure[1])
            leafIdxs .+= N
            # TBD, why permuted hard false here, maybe because tree nodes not leaves?
            bws = [cov(getKernelTree(hode, lidx, false)) for lidx in leafIdxs] 
            # FIXME is a parallel transport needed between different kernel covariances that each exist in different tangent spaces
            mean_bw = Matrix(mean(bws)) # FIXME upgrade to on-manifold mean
            # corrected cov varies from root (only Monte Carlo cov est) to leaves (only selected bandwdith)
            nC = (1 - λ) * (cov(raw_ker)) + λ * mean_bw
            # return a new kernel with cov_continuation, of tree kernel type
            reprT(μ, nC, hode.weights[currIdx]; partial, partl_cb)
        else
            raw_ker
        end
    else
        getKernelLeafAsTreeKer(hode, currIdx, permuted)
    end
end


# TODO ALLOW BOTH BALANCED OR UNBALANCED MASK RETRIEVAL, STARTING WITH FORCED MASK BALANCING
# NOTE, rebalancing reason: deadcenter of covariance is not halfway between points (unconfirmed)
# rebalance if stochastic nearest estimates fall in wrong mask
# see #328 for more details and discussion
function _flipmask_minormax!(smlmask, bigmask, data; argminmax::Function = argmin)
    N = length(smlmask)
    # move minimum mask points over to imask
    for k = 1:((sum(bigmask) - sum(smlmask)) ÷ 2)
        # keep flipping the minimum element from mask into imask set
        # note using first coord, ie.. x-axis as the split axis: `s->s[1]`
        mlis = (1:sum(bigmask))
        ami = argminmax(view(data, bigmask))
        idx = mlis[ami]
        # get idx from orginal list
        flipidx = view(1:N, bigmask)[idx]
        data[flipidx]
        bigmask[flipidx] = xor(bigmask[flipidx], true)
        smlmask[flipidx] = xor(smlmask[flipidx], true)
    end
    return nothing
end

##