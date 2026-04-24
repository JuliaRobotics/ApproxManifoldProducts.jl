


## =========================================================================================
## PCA / Eigen split/sort for tree construction
## =========================================================================================



# covariance eigen decomposition and sort ascending
function eigenCoords!(
    f_CVp::AbstractMatrix;
    partial::Union{Nothing, <:Tuple} = nothing,
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
    partial::Union{Nothing, <:Tuple} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {P <: AbstractArray}
    #
    
    # important, covariance is calculated around mean of points, which enables log to avoid singularities
    # do calculations around mean point on manifold, i.e. towards Riemannian
    p = mean(M, r_PP)
    len = length(r_PP)
    weight = sum(weights)
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

    knl = kernel(p, cv, weight; partial, partl_cb)
    

    r_XXp = log.(Ref(M), Ref(p), r_PP)      # FIXME replace with on-manifold distance
    r_CCp = vee.(Ref(LieAlgebra(M)), r_XXp) # TODO, remove LieGroup/LieAlgebra restriction 

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

    # expecting largest variation on coord dimension `pidx[end]`
    r_R_ax, Λ, pidx = eigenCoords!(cv; partial)
    ax_R_r = r_R_ax'

    # rotate coordinates
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

    # return rotated coordinates and split mask
    return ax_CCp, mask, knl
end


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
    keepbranching = true

    # take a slice of data
    npts = high - low + 1
    idc = low:high
    
    # according to current index permutation (i.e. sort data as you build the tree)
    ido = view(hode.permute, idc)
    
    # secondary recursion termination case, seems odd to have two terminations FIXME
    if npts <= leaf_size
        keepbranching = false
        # HACK mid=-1, knl=nothing if keepbranching false 
        return -1, Set(ido), nothing, keepbranching
    end

    
    _legacybw(s::Nothing) = s
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)
    _kernel_bw = _legacybw(kernel_bw)


    # split the slice of order-permuted data
    _, mask, knl = splitPointsEigen(
        getManifold(hode),
        view(hode.data, ido),
        view(hode.weights, ido);
        kernel,
        kernel_bw = _kernel_bw,
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
    end

    return mid_idx, Set(ido), knl, keepbranching
end
