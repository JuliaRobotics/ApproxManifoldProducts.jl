

# number of data points (aka particles) in tree, i.e. N
Base.length(hode::HomotopyDensity) = Ndim(hode)
Npts(hode::HomotopyDensity) = length(hode.data)
Ndim(hode::HomotopyDensity) = manifold_dimension(getManifold(hode))

getWeights(mt::HomotopyDensity; permute::Bool = true) = permute ? view(mt.weights, mt.permute) : mt.weights

"""
    $SIGNATURES

Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

Notes
- Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
- Second argument controls whether partial dimensions only should be returned (`=true` default).

DevNotes
- Currently converts down to manifold from matrix of coordinates (legacy), to be deprecated TODO
"""
function getPoints(
    hode::HomotopyDensity{partl},
    aspartial::Bool = true;
    permute::Bool = true,
) where {partl}
    #
    pts = permute ? view(hode.data, hode.permute) : hode.data
    # pts = getPoints(x.shim; permute)

    if !aspartial || isnothing(partl)
        # error("MKD getPoints aspartial=true but MKD is not partial")
        return pts
    end

    Mp, Rp, lkup = getManifoldPartial(getManifold(hode), getPartial(hode), pts[1])

    vecP = Vector{typeof(Rp)}(undef, length(pts))
    for (j,pt) in enumerate(pts)
        vecP[j] =  lkup(pt)
    end
    return vecP
end



function getBW(
    x::HomotopyDensity{partl},
    aspartial::Bool = true,
) where {partl}
    bws = getBW.(x.leaf_kernels)
    if isnothing(partl) && aspartial
        return (bw->_getpartial(partl, bw)).(bws)
    end
    return bws
end


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
    mtr::HomotopyDensity{L, M, P, N, HL, HT},
    idx::Int,
    permuted::Bool = false,
) where {M, L, P, N, HL, HT} = convert(HT, getKernelLeaf(mtr, (idx - 1) % N + 1, permuted))

"""
    $SIGNATURES

Return kernel from tree by binary tree index, and convert leaf kernels to tree kernel types if necessary.

Notes:
- BinaryTree (BT) index goes from root=1 to largest leaf 2*N

See also: [`getKernelLeafAsTreeKer`](@ref)
"""
function getKernelTree(
    mtr::HomotopyDensity{L, M, P, N, HL, HT},
    currIdx::Int,
    # must return sorted given name signature "Tree"
    permuted::Bool = false,
    cov_continuation::Bool = false,
) where {M, L, P, N, HL, HT}
    #

    # BinaryTree (BT) index goes from root=1 to largest leaf 2*N
    if currIdx < N
        # cov_continuation correction so that we may build trees with sensible convariance to bandwidth transition from root to leaf
        raw_ker = mtr.tree_kernels[currIdx]
        if cov_continuation
            # depth of this index
            ances_depth = floor(Int, log2(currIdx))
            # how many leaf offsp
            offsp_depth = log2(length(mtr.segments[currIdx]))
            # get approx continuous depth fraction of this index
            λ = (ances_depth) / (ances_depth + offsp_depth)
            # mean bandwidth of all leaf children
            leafIdxs = mtr.segments[currIdx] .|> s -> findfirst(==(s), mtr.permute)
            leafIdxs .+= N
            # TBD, why permuted hard false here, maybe because tree nodes not leaves?
            bws = [cov(getKernelTree(mtr, lidx, false)) for lidx in leafIdxs] 
            # FIXME is a parallel transport needed between different kernel covariances that each exist in different tangent spaces
            mean_bw = Matrix(mean(bws)) # FIXME upgrade to on-manifold mean
            # corrected cov varies from root (only Monte Carlo cov est) to leaves (only selected bandwdith)
            nC = (1 - λ) * (cov(raw_ker)) + λ * mean_bw
            # return a new kernel with cov_continuation, of tree kernel type
            # FIXME, remember partial information
            kernelType = getfield(ApproxManifoldProducts, HT.name.name)
            partial = _getprl(raw_ker)
            μ = mean(raw_ker)
            M_, reprl, partl_cb = getManifoldPartial(getManifold(mtr), _tuple(partial), μ)
            kernelType(μ, nC, mtr.weights[currIdx]; partial, partl_cb)
        else
            raw_ker
        end
    else
        getKernelLeafAsTreeKer(mtr, currIdx, permuted)
    end
end


# check for existence in tree or leaves
function exists_BTLabel(mt::HomotopyDensity{L, M, P, N}, idx::Int) where {M, L, P, N}
    eset = if idx < N
        mt._workaround_isdef_treekernel
    else
        mt._workaround_isdef_leafkernel
    end

    # return existence
    return idx in eset
end

function isLeaf_BTLabel(mt::HomotopyDensity{L, M, P, N}, idx::Int) where {M, L, P, N}
    if exists_BTLabel(mt, leftIndex(mt, idx))
        return false
    elseif exists_BTLabel(mt, rightIndex(mt, idx))
        # TODO likely not needed to check for right child existence
        return false
    else
        return true
    end
end

# check for uniform weights
uniWT(mt::HomotopyDensity) = 1 === length(union(diff(getWeights(mt))))


# check for uniform bandwidths in kernels
function uniBW(mt::HomotopyDensity{L, M, P, N}) where {M, L, P, N}
    if 1 < length(mt.leaf_kernels)
        if !isassigned(mt.leaf_kernels, 1)
            return false
        end
        bw = cov(mt.leaf_kernels[1])
        for lk in view(mt.leaf_kernels, 2:N)
            if !isapprox(bw, cov(lk))
                return false
            end
        end
    end
    return true
end

function Base.show(io::IO, hode::HomotopyDensity{partial, M, P, N, HL, HT}) where {partial, M, P, N, HL, HT}
    printstyled(io, "HomotopyDensity{"; bold = true, color = :blue)
    println(io)
    printstyled(io, "    partial"; bold = true, color = :magenta)
    print(io, " = ", partial, ",")
    println(io)
    printstyled(io, "    M"; bold = true, color = :magenta)
    print(io, " = ", M, ",")
    println(io)
    printstyled(io, "  P  = ", P; color = :magenta)
    println(io)
    printstyled(io, "  N  = ", N; color = :magenta)
    println(io)
    printstyled(io, "  HL = ", HL; color = :magenta)
    println(io)
    printstyled(io, "  HT = ", HT, color = :magenta)
    println(io)
    printstyled(io, "}"; bold = true, color = :blue)
    println(io, "(")
    @assert N == length(hode.data) "show(::HomotopyDensity,) noticed a data size issue, expecting N$(N) == length(.data)$(length(hode.data))"
    if 0 < N
        println(io, "  .data[1:]   :  ", hode.data[1], " ... ", hode.data[end])
        println(io, "  .weights[1:]:  ", hode.weights[1], " ... ", hode.weights[end])
        printstyled(io, "     (uniwt)  :   ", uniWT(hode); color = :light_black)
        println(io)
        print(io, "  .permute[1:]:  ")
        printstyled(io, hode.permute[1], " ... ", hode.permute[end]; color = :light_black)
        println(io)
        print(io, "  .tkernels[") # " __see below__"; color=:light_black)
        if 0 < N
            # printstyled(io, "  .tkernels[1] = "; color=:light_black)
            print(io, "1]:  ")
            printstyled(io, "::HT "; color = :magenta)
            if isassigned(hode.tree_kernels, 1)
                printstyled(io, hode.tree_kernels[1]; color = :light_black)
            else
                printstyled(io, "undef"; color = :red)
                println(io)
            end
            # print(io, "  ...,")
        else
            print(io, "]:   ")
            printstyled(io, "::HT "; color = :magenta)
            println(io)
        end
        printstyled(
            io,
            "     (depth)  :   1+",
            floor(Int, log2(length(hode.tree_kernels)));
            color = :light_black,
        )
        println(io)
        printstyled(io, "     (blncd)  :   ", "true : _wip_"; color = :light_black)
        println(io)
        print(io, "  .lkernels[")
        if 0 < N
            print(io, "1]:  ")
            # printstyled(io, "  .tkernels[1] = "; color=:light_black)
            if isassigned(hode.leaf_kernels, 1)
                printstyled(io, hode.leaf_kernels[1]; color = :light_black)
            else
                printstyled(io, "undef"; color = :red)
                println(io)
            end
            # print(io, "  ...,")
            if 1 < N
                printstyled(io, "         [end]:  "; color = :light_black)
                if isassigned(hode.leaf_kernels, length(hode.leaf_kernels))
                    printstyled(io, hode.leaf_kernels[end]; color = :light_black)
                else
                    printstyled(io, "undef"; color = :red)
                    println(io)
                end
            end
        else
            print(io, "]:   ")
            println(io)
        end
        # printstyled(io, "{1..$N}"; color=:light_black)
        # println(io)
        uBW = uniBW(hode)
        printstyled(io, "     (unibw)  :   ", uBW; color = :light_black)
        println(io)
        if uBW
            printstyled(
                io,
                "         bw   :    ",
                round.((getBW(hode).^2)[1][:]'; digits = 3);
                color = :light_black,
            )
            println(io)
        end
    end
    println(io, ")")
    # TODO ad dmore stats: max depth, widest point, longest chain, max clique size, average nr children

    _round(s::AbstractArray; kw...) = round.(s[:]; kw...)
    _round(s::AbstractVector{<:AbstractMatrix}; kw...) = round.(s[1][:]; kw...)


    println(io, "  Npts:  ", Npts(hode))
    print(io, "  dims:  ", Ndim(hode))
    printstyled(io, isPartial(hode) ? "* --> $(length(getPartial(hode)))" : ""; bold = true)
    println(io)
    println(io, "  prtl:   ", getPartial(hode))
    # bw = (getBW(hode).^2)[1]
    # pvec = isPartial(hode) ? getPartial(hode) : collect(1:length(bw))
    # println(io, "  bws:   ", getBandwidth(hode, true) |> x -> _round(x; digits = 4)) # .|> x->round(x,digits=4))
    println(io, "  ipc:   ", getInfoPerCoord(hode, true) .|> x -> round(x; digits = 4))
    print(io, "   mean: ")
    try
        mn = mean(hode)
        if mn isa ProductRepr # TODO UPDATE to ArrayPartition only, discontinued use of ProductRepr long ago.
            println(io)
            for prt in mn.parts
                println(io, "         ", round.(prt, digits = 4))
            end
        else
            println(io, round.(mn', digits = 4))
        end
    catch
        println(io, "----")
    end
    println(io, ")")

    return nothing
end

Base.show(io::IO, ::MIME"text/plain", hode::HomotopyDensity) = show(io, hode)
function Base.show(io::IO, ::MIME"application/juno.inline", hode::HomotopyDensity)
    return show(io, hode)
end

# covariance eigen decomposition and sort ascending
function eigenCoords!(
    f_CVp::AbstractMatrix;
    partial::Union{Nothing, AbstractVector{<:Integer}} = nothing,
)
    function _decomp(
        evc::AbstractMatrix, 
        evl::AbstractVector, 
        _toflip::Bool = det(evc) < 0
    )
        pidx = _toflip ? sortperm(evl; rev = true) : 1:length(evl)
        Q = evc[:, pidx]
        L = diagm(evl[pidx])
        # FIXME, handle partials -- i.e. embed in larger matrices
        return Q, L, pidx
    end

    # FIXME embed partial dimensions inside the full non-partial covariance.
    _f_CVp = _partialCovToDefault!(partial, _forcemutable(f_CVp))

    E = eigen(_f_CVp)
    f_Q_ax, Λ, pidx = _decomp(E.vectors, E.values)
    # largest variance is on coord `dim = pidx[end]`
    # derotate cloud for easy split
    # swap points order left and right of split
    return f_Q_ax, Λ, pidx
end

function _rotateCoordsPartial(
    M::AbstractLieGroup,
    r_CCp::AbstractVector,
    ax_R_r::AbstractMatrix;
    partial::Union{Nothing, AbstractVector{<:Integer}} = nothing,
)
    _unrollpartial(::Nothing) = LinearAlgebra.I
    _unrollpartial(p::AbstractVector{<:Integer}) = begin
        m = zeros(Int,manifold_dimension(M))
        m[p] .= 1
        return m
    end
    _unrollpartial(p::ArrayPartition) = error("TODO _unrollpartial for ArrayPartition")
    _ = _unrollpartial(partial) # FIXME
    _ax_R_r = _forcemutable(ax_R_r)
    # remove Nans
    for i in axes(_ax_R_r, 1)
        for j in axes(_ax_R_r, 2)
            if !isnothing(partial) && (!(i in partial) || !(j in partial))
                # default values for inactive elements of rotation matrix
                _ax_R_r[i,j] = i == j ? 1.0 : 0.0
            end
            # else leave row and column unchanged
        end
    end

    # rotate coordinates
    return map(r_CCp) do r_Cp
        _r_Cp = _forcemutable(r_Cp)
        for j in 1:length(_r_Cp)
            if !isnothing(partial) && !(j in partial)
                # default values for inactive coordinates
                _r_Cp[j] = 0.0
            end
            # else leave coordinate unchanged
        end
        _ax_R_r * _r_Cp
    end
end


"""
    $SIGNATURES

Give vector of manifold points and split along largest covariance (i.e. major direction)

DevNotes:
- FIXME: upgrade to Manopt version 
  - https://github.com/JuliaRobotics/ApproxManifoldProducts.jl/issues/277
- TODO, instead use Krylov methods (e.g. recursive power series) for next largest eigen vector down depth of tree for efficiency
"""
function splitPointsEigen(
    M::AbstractLieGroup,
    r_PP::AbstractVector{P},
    weights::AbstractVector{<:Real} = ones(length(r_PP)); # FIXME, make static vector unless large
    kernel = ConcentratedGaussianKernel,
    kernel_bw = nothing,
    partial::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {P <: AbstractArray}
    #
    len = length(r_PP)

    # important, covariance is calculated around mean of points, which enables log to avoid singularities
    # do calculations around mean point on manifold, i.e. towards Riemannian
    p = mean(M, r_PP)
    r_XXp = log.(Ref(M), Ref(p), r_PP)      # FIXME replace with on-manifold distance
    r_CCp = vee.(Ref(LieAlgebra(M)), r_XXp) # TODO, remove LieGroup/LieAlgebra restriction 

    D = manifold_dimension(M)
    ndia = ((D - 1) ÷ 2 + 1) * D
    # use provided bandwidth if available, or try estimate multisample covariance
    cv = if ndia < len
        SMatrix{D, D, Float64}(
            Manifolds.cov(M, r_PP; basis = DefaultLieAlgebraOrthogonalBasis()),
        )
    elseif 1 < len <= ndia
        di = diag(Manifolds.cov(M, r_PP; basis = DefaultLieAlgebraOrthogonalBasis()))
        sc = eps(Float64) # maximum(di) 
        SMatrix{D, D, Float64}(
            diagm(di .+ ones(length(di)) * sc),
        )
    else
        SMatrix{D, D, Float64}(zeros(D, D))
    end

    # TODO, handle these if-else cases better
    if isapprox(0.0, norm(cv)) 
        # Fall back case
        bw = if isnothing(kernel_bw)
            @error "Not enough points to estimate covariance" maxlog=5
            # SMatrix{D, D, Float64}(diagm(eps(Float64) * ones(D)))
            cv
        else
            kernel_bw
        end
        return r_CCp, BitVector(ntuple(i -> true, Val(len))), kernel(p, bw)
    end
    # S = SymmetricPositiveDefinite(2)
    # @info "COV" cv LinearAlgebra.isposdef(cv) Manifolds.check_point(S,cv) len
    # expecting largest variation on coord dimension `pidx[end]`
    r_R_ax, Λ, pidx = eigenCoords!(cv; partial)
    ax_R_r = r_R_ax'

    # rotate coordinates
    # ax_CCp = map(r_CCp) do r_Cp
    #     ax_R_r * r_Cp
    # end
    ax_CCp = _rotateCoordsPartial(M, r_CCp, ax_R_r; partial)

    # this is a local test around base point p (not at global 0)
    mask = 0 .<= (ax_CCp .|> (s -> isnothing(partial) ? s[1] : s[partial[1]]))

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

    imask = xor.(mask, true)
    ax_CC1 = (s -> s[1]).(ax_CCp)
    _flipmask_minormax!(imask, mask, ax_CC1; argminmax = argmin)
    _flipmask_minormax!(mask, imask, ax_CC1; argminmax = argmax)

    weight = sum(weights)

    # return rotated coordinates and split mask
    return ax_CCp, mask, kernel(p, cv, weight; partial=_tuple(partial), partl_cb)
end

function buildTree_Manellic!(
    mtree::HomotopyDensity{L, MT, P, N},
    index::Integer, # tree node root=1,left=2n+corr,right=left+1
    low::Integer,   # bottom index of segment
    high::Integer;  # top index of segment;
    kernel = MvNormal,
    kernel_bw = nothing,
    leaf_size = 1,
    partial::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {MT, L, P, N} # FIXME, use just one partial/L
    #
    _legacybw(s::Nothing) = s
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)

    _kernel_bw = _legacybw(kernel_bw)

    # # terminate recursive tree build when all necessary tree kernels have been built
    # if N <= index
    #     return mtree
    # end

    M = getManifold(mtree)
    # take a slice of data
    idc = low:high
    # according to current index permutation (i.e. sort data as you build the tree)
    ido = view(mtree.permute, idc)
    # split the slice of order-permuted data
    ax_CCp, mask, knl = splitPointsEigen(
        M,
        view(mtree.data, ido),
        view(mtree.weights, ido);
        kernel,
        kernel_bw = _kernel_bw,
        partial,
        partl_cb,
    )
    imask = xor.(mask, true)

    # sort the data as 'small' and 'big' elements either side of the eigen split
    big = view(ido, mask)  |> collect
    sml = view(ido, imask) |> collect
    # inplace reorder the slice portion of mtree.permute towards accending
    _ido = SA[sml...; big...]
    # ido .= SA[sml...; big...]
    for (i,v) in enumerate(_ido)
        ido[i] = v
    end

    # terminate recursive tree build when all necessary tree kernels have been built
    if N <= index
        return mtree
    end

    npts = high - low + 1
    mid_idx = low + sum(imask) - 1

    lft = mid_idx <= low ? low : leftIndex(mtree, index)
    rgt = high <= mid_idx + 1 ? high : rightIndex(mtree, index)

    if leaf_size < npts
        if lft != low # mid_idx
            # recursively call two branches of tree, left
            buildTree_Manellic!(
                mtree,
                lft,
                low,
                mid_idx;
                kernel,
                kernel_bw = _kernel_bw,
                leaf_size,
                partial,
                partl_cb,
            )
        end
        if rgt != high
            # and right subtree
            buildTree_Manellic!(
                mtree,
                rgt,
                mid_idx + 1,
                high;
                kernel,
                kernel_bw = _kernel_bw,
                leaf_size,
                partial,
                partl_cb,
            )
        end
    end

    if index < N
        tkT = eltype(mtree.tree_kernels)
        # TBD, maybe a constructor instead?
        _knl = tkT(knl; partl_cb)
        # _knl = convert(tkT, knl)
        # set tree kernel
        mtree.tree_kernels[index] = _knl
        push!(mtree._workaround_isdef_treekernel, index)
        mtree.segments[index] = Set(ido)
    end

    return mtree
end

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
    partial::Union{Nothing, AbstractVector{<:Integer}, <:Tuple} = nothing,
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
    lkern = SizedVector{N, lknlT}(undef)
    _workaround_isdef_leafkernel = Set{Int}()
    for i = 1:N
        nkr = kernel(r_PP[i], lCV; partial = _tuple(partial), partl_cb=prlcb)
        lkern[i] = nkr
        push!(_workaround_isdef_leafkernel, i + N)
    end

    _hode = ApproxManifoldProducts.HomotopyDensity{
        _tuple(partial),
    }(;
        manifold = M,
        data = r_PP,
        weights,
        leaf_kernels = lkern,                           # leaf_kernels
        tree_kernels = SizedVector{N, tknlT}(undef),    # tree_kernels
        _workaround_isdef_leafkernel,
    );

    #
    tosort_leaves = buildTree_Manellic!(
        _hode,
        1, # start at root
        1, # spanning all data
        N; # to end of data
        kernel,
        kernel_bw,
        partial,
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
    lkern = SizedVector{N, KL}(undef)
    _workaround_isdef_leafkernel = Set{Int}()
    for i = 1:N
        r_PP[i] = mean(r_ker[i])
        lkern[i] = if isnothing(kernel_bw)
            r_ker[i]
        else
            updateKernelBW(r_ker[i], kernel_bw) # TODO handle vector of kernel_bws
        end
        push!(_workaround_isdef_leafkernel, i + N)
    end

    mtree = HomotopyDensity{
        _getprl(r_ker[1]),
    }(;
        manifold = M,
        data = r_PP,
        weights,
        permute = MVector{N, Int}(1:N),
        leaf_kernels = lkern,
        tree_kernels = SizedVector{N, KT}(undef),
        segments = SizedVector{N, Set{Int}}(undef),
        _workaround_isdef_leafkernel,
        _workaround_isdef_treekernel = Set{Int}(),
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

function updateBandwidths(
    hode::HomotopyDensity{L, M, P, N, HL}, 
    bws;
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {L, M, P, N, HL}
    #
    _getBW(s::Float64, ::Int) = [s;;]
    _getBW(s::AbstractVector{<:Real}, ::Int) = s
    _getBW(s::AbstractMatrix{<:Real}, ::Int) = s
    _getBW(s::AbstractVector{<:AbstractArray}, _i::Int) = s[_i]

    _leaf_kernels = SizedVector{N, HL}(undef)
    for (i, lk) in enumerate(hode.leaf_kernels)
        nkl = ConcentratedGaussianKernel(lk; Σ = _getBW(bws, i), partl_cb)
        _leaf_kernels[i] = nkl # updateKernelBW(lk, _getBW(bws, i))
    end
    return HomotopyDensity{
        L,
    }(
        manifold = getManifold(hode),
        data = hode.data,
        weights = hode.weights,
        permute = hode.permute,
        leaf_kernels = _leaf_kernels,
        tree_kernels = hode.tree_kernels,
        segments = hode.segments,
        _workaround_isdef_leafkernel = hode._workaround_isdef_leafkernel,
        _workaround_isdef_treekernel = hode._workaround_isdef_treekernel,
    )
end

"""
    $SIGNATURES
    
For Manellic tree parent kernels, what is the 'smallest' and 'biggest' covariance.

Notes:
- Thought about `det` for covariance volume but long access of pancake (smaller volume) is not minimum compared to circular covariance. 
"""
function getBandwidthSearchBounds(mtree::HomotopyDensity)
    upper = cov(mtree.tree_kernels[1])

    #FIXME isdefined does not work as expected for mtree.tree_kernels, so using length-1 for now
    # this will break if number of points is not a power of 2. 
    
    lower_diag = diag(cov(mtree.tree_kernels[1]))
    for i in 2:(length(mtree.tree_kernels) - 1)
        # FIXME use consolidated getKernelTree instead
        if isassigned(mtree.tree_kernels, i)
            hdg = hcat(lower_diag, diag(cov(mtree.tree_kernels[i])))
            lower_diag = minimum(hdg; dims = 2)
        end
        # lower_diag = minimum(hcat(lower_diag, diag(cov(mtree.tree_kernels[i]))); dims = 2)
    end

    # floors make us feel safe, but hurt when faceplanting
    lower_diag = maximum(hcat(lower_diag, 1e-8 * ones(length(lower_diag))); dims = 2)[:]

    # Give back lower as diagonal only covariance matrix
    lower = diagm(lower_diag)

    return lower, upper
end

"""
    $SIGNATURES

Evaluate the belief density for a given Manellic tree.

DevNotes:
- Computational Geometry
  - use geometric computing for faster evaluation
- Dual tree evaluations
  - Holmes, M.P., Gray, A.G. and Isbell Jr, C.L., 2010. Fast kernel conditional density estimation: A dual-tree Monte Carlo approach. Computational statistics & data analysis, 54(7), pp.1707-1718.
  - Curtin, R., March, W., Ram, P., Anderson, D., Gray, A. and Isbell, C., 2013, May. Tree-independent dual-tree algorithms. In International Conference on Machine Learning (pp. 1435-1443). PMLR.
- Fast kernels
- Parallel transport shortcuts?
"""
function evaluate(
    hode::HomotopyDensity{partl},
    pt,
    LOO::Bool = false,
    force_kbw = nothing,
) where {partl}
    # # force function barrier, just to be sure dyndispatch is limited
    # _F() = getfield(ApproxManifoldProducts,HL.name.name)
    # _F_ = _F() 

    pts = getPoints(hode, false)
    w = getWeights(hode)

    # isapprox uses partial version
    M_, reprl, cb = getManifoldPartial(getManifold(hode), partl)

    sumval = 0.0
    # FIXME, brute force for loop
    for (i, t) in enumerate(pts)
        if !LOO || !isapprox(M_, cb(pt), cb(t))
        # if !LOO || !isapprox(getManifold(hode), pt, t)
            # TBD, is this assuming length(pts) and length(hode.leaf_kernels) are the same?
            # FIXME use consolidated getKernelLeaf instead
            ekr = hode.leaf_kernels[i]
            ekr = updateKernelBW(ekr, force_kbw)
            # remember special handling for partials via ekr itself
            oneval = hode.weights[i] * evaluate(getManifold(hode), ekr, pt)
            # leave one out requires kernel weighting to removal of leave out weight
            oneval *= !LOO ? 1 : 1 / (1 - w[i])
            sumval += oneval
        end
    end

    return sumval
end

"""
    $SIGNATURES

Return vector of weights of evaluated proposal label points against density.

DevNotes:
- TODO should evat points be of equal weights?  If multiscale sampling goes down unbalanced trees?
- FIXME how should partials be handled here? 
- FIXME, use multipoint evaluation such as NN (not just one point at a time)
"""
function evaluateDensityAtPoints(
    M::AbstractManifold,
    density,
    eval_at_points,
    normalize::Bool = false,
)
    # evaluate new sampling weights of points in out component
    # TODO use agnostic-Dual tree or MonteCarloDualTree evaluation
    # vector for storing resulting weights
    smw = zeros(length(eval_at_points))
    for (i, ev) in enumerate(eval_at_points)
        # single kernel evaluation
        smw[i] = evaluate(M, density, ev)
        # δc = distanceMalahanobisCoordinates(M,tmp_product,ev)
    end

    # Note convenience only
    if normalize
        _s = sum(smw)
        if isapprox(_s, 0.0)
            #assume L'Hopital or similar
            smw .= 1 / length(smw)
        else
            smw ./= _s
        end
    end

    # return weights
    return smw
end

function expectedLogL(
    mt::HomotopyDensity,
    epts::AbstractVector,
    LOO::Bool = false,
    force_kbw = nothing,
)
    T = Float64
    # TODO really slow brute force evaluation, use agnostic-DualTree or MonteCarloDualTree
    eL = MVector{length(epts), T}(undef)
    for (i, p) in enumerate(epts)
        # LOO skip for leave-one-out
        eL[i] = evaluate(mt, p, LOO, force_kbw)
    end
    # set numerical tolerance floor
    zrs = findall(isapprox.(0, eL))
    # nominal case with usable evaluation points
    eL[zrs] .= 1.0

    # weight and return within numerical reach
    w = getWeights(mt)
    if any(0 .!= w[zrs])
        -Inf
    else
        w' * (log.(eL))
        # return mean(log.(eL)) #?
    end
end

function entropy(hode::HomotopyDensity, force_kbw = nothing)
    return -expectedLogL(hode, getPoints(hode, false), true, force_kbw)
end

(hode::HomotopyDensity)(evalpt::AbstractArray) = evaluate(hode, evalpt)

"""
    $SIGNATURES

Calculate one product of proposal kernels, as defined  BTLabels.
"""
function calcProductKernelBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector,
    labels_sampled::AbstractVector{<:Integer},
    looidx::Union{Int, Nothing} = nothing,
    propIdxs_Gibbs::AbstractVector{<:Integer} = 1:length(proposals);
    permute::Bool = true, # true because signature is BTLabels
    weight::Real = 1.0,
)
    # select a density label from the other proposals
    prop_and_label = Tuple{Int, Int}[]
    for s in setdiff(propIdxs_Gibbs, isnothing(looidx) ? Int[] : Int[looidx;])
        # tuple of which leave-one-out-proposal and its new latest label selection
        push!(prop_and_label, (s, labels_sampled[s]))
    end
    # get raw kernels from tree, also as tree_kernel type
    # TODO COVARIANCE CONTINUATION CORRECTION FOR DEPTH OF TREE KERNELS
    components = map(
        pr_lb -> getKernelTree(proposals[pr_lb[1]], pr_lb[2], permute, true),
        prop_and_label,
    )

    # TODO upgrade to tuples
    return calcProductGaussians(M, [components...]; weight)
end

function calcProductKernelsBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector,
    N_lbl_sets::AbstractVector{<:NTuple},
    permute::Bool = true; # true because signature is BTLabels
    weights = 1 / length(N_lbl_sets) .* ones(length(N_lbl_sets)),
)
    #
    # partials = getKernelTree.(proposals, Ref(1)) .|> _getprl
    # @show _mergepartials(M, partials)
    # T = typeof(getKernelTree(proposals[1], 1)) # FIXME FIXME FIXME for products of partials, not just [1]
    N = length(N_lbl_sets)
    # FIXME abstract vectorT not type-stable
    post = Vector{ConcentratedGaussianKernel}(undef, N) 

    for (i, lbs) in enumerate(N_lbl_sets)
        post[i] = calcProductKernelBTLabels(M, proposals, _makevec(lbs); permute, weight = weights[i])
    end

    return post
end


# TODO why not use a standardized `getChildren`?
function generateLabelPoolRecursive(
    proposals::AbstractVector{<:HomotopyDensity},
    labels_sampled::AbstractVector{<:Integer},
)
    # NOTE at top of tree, selections will be [1,1]
    child_label_pools = Vector{Vector{Int}}()

    # Are all selected labels leaves?
    all_leaves = true
    for _ = 1:length(proposals)
        push!(child_label_pools, Vector{Int}())
    end
    for (o, sel) in enumerate(labels_sampled)
        isleaf = true
        # add interval of left and right children for next scale label sampling
        if exists_BTLabel(proposals[o], leftIndex(proposals[o], sel))
            push!(child_label_pools[o], leftIndex(proposals[o], sel))
            isleaf = false
        end
        if exists_BTLabel(proposals[o], rightIndex(proposals[o], sel))
            push!(child_label_pools[o], rightIndex(proposals[o], sel))
            isleaf = false
        end
        all_leaves &= isleaf
        if isleaf
            push!(child_label_pools[o], sel)
        end
    end

    return child_label_pools, all_leaves
end

"""
    $SIGNATURES

Notes:
- Advise, 2<=MC to ensure multiscale works during decent transitions (TBD obsolete requirement)
- To force sequential Gibbs on leaves only, use:
  `label_pools = [[(length(getPoints(prop))+1):(2*length(getPoints(prop)));] for prop in proposals]`
- Taken from: Sudderth, E.B., Ihler, A.T., Isard, M., Freeman, W.T. and Willsky, A.S., 2010. 
  Nonparametric belief propagation. Communications of the ACM, 53(10), pp.95-103.
"""
function sampleProductSeqGibbsBTLabel(
    M::AbstractManifold,
    proposals::AbstractVector{<:HomotopyDensity},
    MC::Int = 3,
    # pool of sampleable labels
    label_pools::Vector{Vector{Int}} = [[1:1;] for _ in proposals],
    labels_sampled::Vector{Int} = [rand(label_pools[i]) for i in 1:length(proposals)];
    # multiscale_parents = nothing;
    MAX_RECURSE_DEPTH::Int = 24, # 2^24 is so deep
    _labelsChoosen::Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}} = Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}()
)
    # local helpers for partials either vec or nothing
    _leng(s::Nothing) = 0
    _leng(s::Union{<:AbstractVector{<:Integer}, <:Tuple}) = length(s)

    # apply further partials to existing kernel
    # how many incoming proposals
    d = length(proposals)
    propIdxs_Gibbs = 1:d

    _trivial_label_pool = all(length.(label_pools) .== 1)
    # pick the next leave-out proposal
    # TODO, gibbSeq might be different for unbalanced nodes "cross-products" during multiscale
    for _burn = 1:MC, lvout_idx in propIdxs_Gibbs
        # on first pass labels_sampled come from parent-recursive as part of multi-scale (i.e. pre-homotopy) operations
        # calc product of Gaussians from currently selected \LOO-proposals
        lvin_product_tmp = calcProductKernelBTLabels(
            M,
            proposals,
            labels_sampled,
            lvout_idx,
            propIdxs_Gibbs;
            permute = false,
        )
        
        # to find leave-out (LO) resample weights, evaluate leave-in (LI) mean against temporary leavein_product kernel
        lvout_centers = [mean(getKernelTree(proposals[lvout_idx], i, false)) for i in label_pools[lvout_idx]]
        # if lvout_centers are partial, then only evaluate with partial lvin_product_tmp
        lvout_prl = _getprl(getKernelTree(proposals[lvout_idx], label_pools[lvout_idx][1], false))
        lvin_product_tmp_partial = _intersectpartials(M, lvin_product_tmp, lvout_prl)

        # overcome case where no partial overlap exists
        resample_weights = if 0 < _leng(_getprl(lvin_product_tmp_partial))
            resample_weights = evaluateDensityAtPoints(M, lvin_product_tmp_partial, lvout_centers, true)
            # update label-distribution of out-proposal from product of selected LOO-proposal components
            p = Categorical(resample_weights)
            labels_sampled[lvout_idx] = label_pools[lvout_idx][rand(p)]
            resample_weights
        else
            NaN*ones(length(lvout_centers))
        end

        # slightly heavy memory usage to aid DX
        push!(_labelsChoosen, (;
            loo = lvout_idx,
            selected = deepcopy(labels_sampled),
            pool = deepcopy(label_pools),
            catp = deepcopy(resample_weights),
        ))

        # don't have to resample if only one label to choose from
        if _trivial_label_pool && ( lvout_idx == propIdxs_Gibbs[end])
            break
        end
    end

    # construct new label pool for children in multiscale
    child_label_pools, all_leaves = generateLabelPoolRecursive(proposals, labels_sampled)

    # recursively call sampling down the multiscale tree ("pyramid") -- aka homotopy
    # limit recursion to MAX_RECURSE_DEPTH
    # FIXME, final label selection should not be sensitive to being all_leaves.
    if 0 < MAX_RECURSE_DEPTH && !all_leaves
        # @info "Recurse down manellic tree for multiscale product"
        # labels_sampled_copy = deepcopy(labels_sampled)
        labels_sampled = sampleProductSeqGibbsBTLabel(
            M,
            proposals,
            MC,
            child_label_pools;
            # labels_sampled_copy; # randomly sample from new child pool
            MAX_RECURSE_DEPTH = MAX_RECURSE_DEPTH - 1,
            _labelsChoosen,
        )

        # TODO, [circa 2006, Rudoy & Wolfe] detailed balance (Hastings) by rejecting a multiscale decent given simulated or parallel tempering
        # recursive call of sampleProductSeqGibbsBTLabel but with same parameters as this function invokation, aka reject the decend
    end

    #
    return labels_sampled
end


function sampleProductSeqGibbsBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector{<:HomotopyDensity},
    MC::Int = 3,
    N::Int = round(Int, mean(Npts.(proposals))), # FIXME use getLength or length of proposal (not getPoints)
    label_pools = [[1:1;] for _ in proposals];
    _labelsChoosen_pp::Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}} = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)
)
    #
    d = length(proposals)
    posterior_labels = Vector{NTuple{d, Int}}(undef, N)

    for i = 1:N
        _labelsChoosen_pp[i] = Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}()
        posterior_labels[i] =
            tuple(sampleProductSeqGibbsBTLabel(M, proposals, MC, label_pools; _labelsChoosen = _labelsChoosen_pp[i])...)
    end

    return posterior_labels
end

##
