


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
    r_PP::AbstractVector{P};
    # weights::AbstractVector{<:Real} = ones(length(r_PP)); # FIXME, make static vector unless large
    # kernel = ConcentratedGaussianKernel,
    kernel_bw = nothing,
    partial::Union{Nothing, <:Tuple} = nothing,
    # partl_cb::Union{Nothing, <:Function} = nothing,
) where {P <: AbstractArray}
    #
    _legacybw(s::Nothing) = s
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)

    # important, covariance is calculated around mean of points, which enables log to avoid singularities
    # do calculations around mean point on manifold, i.e. towards Riemannian
    len = length(r_PP)
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

    # handle some edge cases relating to covariance estimation
    bw = if isapprox(0.0, norm(cv)) 
        # Fall back case
        if isnothing(kernel_bw)
            error("Provided data points have no measurable covariance and no kernel bandwidth was provided.")
        else
            _legacybw(kernel_bw)
        end
    else
        cv
    end

    p = mean(M, r_PP)
    r_XXp = log.(Ref(M), Ref(p), r_PP)      # FIXME replace with on-manifold distance
    r_CCp = vee.(Ref(LieAlgebra(M)), r_XXp) # TODO, remove LieGroup/LieAlgebra restriction 

    # default return values
    #     weight = sum(weights)
    # knl = kernel(p, bw, weight; partial, partl_cb)
    mask = BitVector(ntuple(i -> true, Val(len)))
    # geometric split made possible by sum(imask) instead of just data split (classification labeling must happen in cosort) 
    midoffset = sum(xor.(mask, true)) - 1
    ax_CCp = r_CCp

    # TODO, handle these if-else cases better
    if !isapprox(0.0, norm(cv)) 
        # NOTE, this if block started out with coordinates only, so `partial` while ignoring `partl_cb`.
        # expecting largest variation on coord dimension `pidx[end]`
        r_R_ax, Λ, pidx = eigenCoords!(cv; partial)
        ax_R_r = r_R_ax'

        # rotate coordinates
        ax_CCp = _rotateCoordsPartial(M, r_CCp, ax_R_r; partial)

        # this is a local test around base point p (not at global 0)
        mask = 0 .<= (ax_CCp .|> (s -> isnothing(partial) ? s[1] : s[partial[1]]))

        imask = xor.(mask, true)
        ax_CC1 = (s -> s[1]).(ax_CCp)
        _flipmask_minormax!(imask, mask, ax_CC1; argminmax = argmin)
        _flipmask_minormax!(mask, imask, ax_CC1; argminmax = argmax)

        # geometric split made possible by sum(imask) instead of just data split (classification labeling must happen in cosort) 
        midoffset = sum(xor.(mask, true)) - 1
    end

    # return rotated coordinates and split mask
    return ax_CCp, mask, midoffset, p, bw
end


