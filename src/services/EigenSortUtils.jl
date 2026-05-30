


## =========================================================================================
## PCA / Eigen split/sort for tree construction
## =========================================================================================



# covariance eigen decomposition and sort ascending
function eigenCoords!(
    f_CVp::AbstractMatrix;
    partial::Union{Nothing, <:Tuple} = nothing,
    kernel_bw = nothing,
)
    _legacybw(b::Nothing, c::AbstractMatrix) = c
    _legacybw(b::AbstractMatrix, c::AbstractMatrix) = isapprox(0.0, norm(c)) ? b : c
    _legacybw(b::AbstractVector, c::AbstractMatrix) = isapprox(0.0, norm(c)) ? diagm(b) : c

    # TBD embed partial dimensions inside the full non-partial covariance.
    # _f_CVp mutability required during rank deficient fix later 
    _f_CVp = _partialCovToDefault!(partial, _forcemutable(f_CVp))
    # _f_CVp = _partialCovToDefault!(partial, _forcemutable(_legacybw(kernel_bw, f_CVp)))

    # workaround for zero covariance -- TBD rather remove
    if isapprox(0.0, norm(f_CVp))
        len = isnothing(partial) ? size(f_CVp, 1) : length(partial)
        f_Q_ax = Matrix{Float64}(I, len, len)
        return f_Q_ax, _legacybw(kernel_bw, _f_CVp)
    end

    # towards top-down bandwidth continuation
    # perform eigend decomposition and reconstruction on only the active dimensions
    _partlCVinpl = _viewprl(_f_CVp, partial)
    _evv = eigen(_partlCVinpl)
    # HomotopyDensity tree build error, bandwidth $bw is not a valid covariance matrix for MvNormal kernel
    # Reconstruct to nearest positive definite matrix using Eigen factorization
    _evv_vals = _forcemutable(_evv.values)
    # Ensure the returned bandwidth/covariance matrix is positive definite by small increases in zero eigen values
    # more likely to effect small sample sizes
    _evv_vals[_evv_vals .<= 1e-14] .= 1e-14
    # in-place reconstruct covariance matrix with the modified eigenvalues
    _partlCVinpl .= _evv.vectors * diagm(_evv_vals) * _evv.vectors'


    # _evv2 = eigen(_f_CVp) # FIXME, now includes Inf and repeat calc barr partials
    # TBD, why sort pidx on negetive determinant? TODO write motive -- something about largest eigen value at pidx[end]
    pidx = det(_evv.vectors) < 0 ? sortperm(_evv.values; rev = true) : collect(1:length(_evv.values))
    f_Q_ax = _evv.vectors[:, pidx]
    ## FIXME, only do one eigen w partials

    # largest variance is on coord `dim = pidx[end]`
    # derotate cloud for easy split
    # swap points order left and right of split

    # Eigen rotation matrix, TODO likely easier to replace wholesale with SVD instead
    if isapprox(0.0, norm(f_CVp))
        len = isnothing(partial) ? size(f_CVp, 1) : length(partial)
        f_Q_ax = Matrix{Float64}(I, len, len)
    end

    return f_Q_ax, _legacybw(kernel_bw, _f_CVp)
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
        r_R_ax, bw = eigenCoords!(cv; partial, kernel_bw)
        ax_R_r = r_R_ax'

        # rotate coordinates
        ax_CCp = _rotateCoordsPartial(M, r_CCp, ax_R_r; partial)

        # Sort data along the major eigen vector direction -- i.e. first coord after rotation
        #  this is a local test around base point p (not at global 0)
        ax_CC1 = (s -> s[1]).(ax_CCp)
        mask = 0 .<= ax_CC1
        # mask = 0 .<= (ax_CCp .|> (s -> isnothing(partial) ? s[1] : s[partial[1]]))

        imask = xor.(mask, true)
        _flipmask_minormax!(imask, mask, ax_CC1; argminmax = argmin)
        _flipmask_minormax!(mask, imask, ax_CC1; argminmax = argmax)

        # geometric split made possible by sum(imask) instead of just data split (classification labeling must happen in cosort) 
    # end
    midoffset = sum(xor.(mask, true)) - 1


    # return rotated coordinates and split mask
    return mask, midoffset, p, bw
end


