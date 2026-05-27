


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
        pidx = _toflip ? sortperm(evl; rev = true) : collect(1:length(evl))
        Q = evc[:, pidx]
        L = diagm(evl[pidx])
        # FIXME, handle partials -- i.e. embed in larger matrices
        return Q, L, pidx
    end

    # workaround for zero covariance
    if isapprox(0.0, norm(f_CVp))
        len = isnothing(partial) ? size(f_CVp, 1) : length(partial)
        f_Q_ax = Matrix{Float64}(I, len, len)
        Λ = zeros(len, len)
        pidx = collect(1:len)
        return f_Q_ax, Λ, pidx
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
    kernel_bw = nothing,
    partial::Union{Nothing, <:Tuple} = nothing,
    # partl_cb::Union{Nothing, <:Function} = nothing,
) where {P <: AbstractArray}
    #
    _legacybw(b::Nothing, c::AbstractMatrix) = c
    _legacybw(b::AbstractMatrix, c::AbstractMatrix) = isapprox(0.0, norm(c)) ? b : c
    _legacybw(b::AbstractVector, c::AbstractMatrix) = isapprox(0.0, norm(c)) ? diagm(b) : c

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

    p = mean(M, r_PP)
    r_XXp = log.(Ref(M), Ref(p), r_PP)      # FIXME replace with on-manifold distance
    r_CCp = vee.(Ref(LieAlgebra(M)), r_XXp) # TODO, remove LieGroup/LieAlgebra restriction 

    # default return values
    mask = BitVector(ntuple(i -> true, Val(len)))
    # geometric split made possible by sum(imask) instead of just data split (classification labeling must happen in cosort) 
    ax_CCp = r_CCp

    # # TODO, handle these if-else cases better
    # if !isapprox(0.0, norm(cv)) 
        # NOTE, this if block started out with coordinates only, so `partial` while ignoring `partl_cb`.
        # expecting largest variation on coord dimension `pidx[end]`
        r_R_ax, Λ, pidx = eigenCoords!(cv; partial)
        ax_R_r = r_R_ax'

        # rotate coordinates
        ax_CCp = _rotateCoordsPartial(M, r_CCp, ax_R_r; partial)

        # Sort data along the major eigen vector direction -- i.e. first coord after rotation
        #  this is a local test around base point p (not at global 0)
        mask = 0 .<= (ax_CCp .|> (s -> isnothing(partial) ? s[1] : s[partial[1]]))

        imask = xor.(mask, true)
        ax_CC1 = (s -> s[1]).(ax_CCp)
        _flipmask_minormax!(imask, mask, ax_CC1; argminmax = argmin)
        _flipmask_minormax!(mask, imask, ax_CC1; argminmax = argmax)

        # geometric split made possible by sum(imask) instead of just data split (classification labeling must happen in cosort) 
    # end
    midoffset = sum(xor.(mask, true)) - 1

    # handle some edge cases relating to covariance estimation
    bw = _legacybw(kernel_bw, cv)

    # towards top-down bandwidth continuation
    # pick npts 3 because non-posdef issue more likely for small leaves
    lp = length(r_PP)
    mbw = _forcemutable(bw)
    if lp <= 4
        _pbw = _viewprl(mbw, partial)
        _evv = eigen(_pbw)
        if sum(_evv.values .> 1e-14) < length(_evv.values)
            # HomotopyDensity tree build error, bandwidth $bw is not a valid covariance matrix for MvNormal kernel
            # Reconstruct to nearest positive definite matrix using Eigen factorization
            _evv_vals = _forcemutable(_evv.values)
            _evv_vals[_evv_vals .<= 1e-14] .= 1e-14
            # in-place reconstruct covariance matrix with the modified eigenvalues
            _pbw .= _evv.vectors * diagm(_evv_vals) * _evv.vectors'
        end
    end
    # return rotated coordinates and split mask
    return ax_CCp, mask, midoffset, p, mbw
end


