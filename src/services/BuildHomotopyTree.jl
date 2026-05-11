
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
    manif::M,
    r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
    N = length(r_PP),
    weights::AbstractVector{<:Real} = ones(N) .* (1 / N),
    kernel = ConcentratedGaussianKernel,
    kernel_bw = nothing, # TODO
    partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {M <: AbstractManifold, P <: AbstractArray}
    #
    
    D = manifold_dimension(manif)
    CV = SMatrix{D, D, Float64, D * D}(diagm(ones(D)))
    prlcb = if isnothing(partl_cb) && !isnothing(partial)
        M_, reprl, cb = getManifoldPartial(manif, partial)
        cb
    else
        partl_cb
    end
    
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)
    _legacybw(::Nothing) = CV
        
    lCV = _legacybw(kernel_bw)
    tknlT = kernel(r_PP[1], CV; partial=_tuple(partial), partl_cb=prlcb) |> typeof
    lknlT = kernel(r_PP[1], lCV; partial = _tuple(partial), partl_cb=prlcb) |> typeof

    # leaf kernels
    lkern = Vector{lknlT}(undef, N)
    for i = 1:N
        nkr = kernel(r_PP[i], lCV; partial = _tuple(partial), partl_cb=prlcb)
        lkern[i] = nkr
    end
    tkern = Vector{tknlT}(undef, N)

    _partial = _tuple(partial)
    reprkind = HomotopyRepr(;
        topologykind = BinaryTruncFixedDepth{3}(),
        reprkind = ConcentratedGaussianKernel(),
        statekind = manif,
        partial = _partial,
    )

    # TODO consolidate w legacy kernel_bw
    d = manifold_dimension(getManifold(reprkind))
    trailing_forms = SparseArrays.sparsevec(Dict(
        1 => SMatrix{d,d,Float64}(cov(lkern[1])),
    ), 1) # assume size 1 during refactor -- i.e. universal bandwidth at leaves

    _hode = HomotopyDensityLive{
        typeof(reprkind),
        eltype(r_PP),
        eltype(r_PP),
        Matrix{Float64},
        eltype(trailing_forms),
    }(;
        reprkind,
        points = r_PP,
        weights,
        trailing_forms,
    )

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

    return tosort_leaves
end



function buildTree_Manellic!(
    manif::M,
    r_ker::AbstractVector{KL}; # vector of points referenced to the r_frame
    N = length(r_ker),
    weights::AbstractVector{<:Real} = ones(N) .* (1 / N),
    kernel = KL,
    kernel_bw = nothing, # TODO
    # partial = ??? TBD -- it should already be in the kernels
) where {M <: AbstractManifold, KL <: ConcentratedGaussianKernel}
    #
    _μT() = typeof(mean(r_ker[1]))
    D = manifold_dimension(manif)
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

    trailing_forms = SparseArrays.sparsevec(Dict(
        1 => SMatrix{D,D}(cov(lkern[1])),
    ), 1)

    partial = _getprl(r_ker[1])
    mtree = HomotopyDensity_legacy(;
        partial,
        manifold = manif,
        points = r_PP,
        weights,
        trailing_forms,
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
    mid_idx, sml, big = truncateOrSplitsort!(
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
    if 0 < length(sml)
        lftidx = leftIndex(hode, index)
        hode.structure[lftidx] = (sml)
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
        sml = hode.structure[lftidx] # update sml since tree build will continue
    end

    # recursively check need for right subtree
    if 0 < length(big)
        rhtidx = rightIndex(hode, index)
        hode.structure[rhtidx] = (big)
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
        big = hode.structure[rhtidx] # update big since tree build will continue
    end

    # at start, .structure[1] is spread over all data 1:N and populated by HomotopyDensity constructor
    # vcat ensures sml, big are buffered for in-place "swap" of the slice portion of structure, else points overwritten prematurely
    # TODO, not sure if this step is needed or if this is the right place for this call
    # must set once for each node, above each child node was permuted and now the parent list must also be permuted so that sorting propagates to top list
    hode.structure[index] .= vcat(sml, big)

    return hode
end


## ================================================================================
## Binary tree split and sort
## ================================================================================


function truncateOrSplitsort!(
    hode::HomotopyDensity,
    low::Integer,
    high::Integer,
    index::Integer;
    leaf_size::Integer = 1,
    kernel_bw = nothing,
    partial::Union{Nothing, <:Tuple},
    partl_cb::Union{Nothing, <:Function},
)
    # recursion termination case
    # geometric split instead of data split (must happen in cosort classification labeling) 
    # TBD, untested leaf_size is not 1
    npts = high - low + 1
    if (npts <= leaf_size)
        return -1, Int[], Int[]
    end

    # according to current index permutation (i.e. sort data as you build the tree root to leaves)
    idxsubset = hode.structure[index]
        # reminder which slice of permuteidxs to use
        # idc = low:high

    midoffset, sml, big, p, bw = splitsortBinary!(
        hode,
        idxsubset;
        kernel_bw,
        # partial, # TODO remove, get from hode internally
    )

    # TRUNCATION CRITERIA, only happens for majors
    # TBD, possible location for populating .majors_ fields here...?
    # TBD, this part will likely be refactored with `.majors_*` fields
    # store tree kernel and segment indices; after sorting
    if (leaf_size < npts) && (index <= Npts(hode))
        # set tree kernel
        # NOTE, THIS USED TO BE AFTER recursive subtree build
        wei = sum(view(hode.weights, idxsubset))
        knl = ConcentratedGaussianKernel(
            p, bw, wei; # TODO, try drop need for p here
            partial, partl_cb
        )
        # NEW, set majors_ fields here
        if length(hode.principal_coeffs) < index
            resize!(hode.principal_coeffs, index)
            resize!(hode.principal_elements, index) 
            resize!(hode.principal_forms, index)
        end
        hode.principal_coeffs[index] = sum(view(hode.weights, idxsubset))
        hode.principal_elements[index] = mean(knl)
        hode.principal_forms[index] = cov(knl)
    end

    # for binary split
    mid_idx = low + midoffset

    return mid_idx, sml, big
end



function splitsortBinary!(
    hode::HomotopyDensity,
    idxsubset::AbstractVector{<:Integer};
    kernel_bw = nothing,
    # partial::Union{Nothing, <:Tuple},
)
    # split the slice of order-permuted data
    _, mask, midoffset, p, bw = splitPointsEigen(
        getManifold(hode),
        view(hode.points, idxsubset);
        kernel_bw,
        partial = getPartial(hode),
    )
    imask = xor.(mask, true)
    
    if 2 < abs(sum(mask) - sum(imask))
        error("DX bug stop, eigen split not well defined for two population groups.")
    end

    # sort the data as 'small' and 'big' points either side of the eigen split
    # towards accending (in-place) reorder of the slice portion
    # in-place replacement requires a temporary buffer -- achieved by vcat, else can use collect here
    big = view(idxsubset, mask)  |> collect
    sml = view(idxsubset, imask) |> collect
    # TODO, reduce mem with idxsubset[1:nsml] .= sml ... instead :::: vcat buffers points for in-place "swap", else points overwritten prematurely 
    idxsubset .= vcat(sml, big)  

    return midoffset, collect(sml), collect(big), p, bw
end

