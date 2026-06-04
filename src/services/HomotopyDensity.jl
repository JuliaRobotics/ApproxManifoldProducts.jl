

## ==========================================================================================
## HomotopyDensity constructorhelper functions
## ==========================================================================================


function DistributedFactorGraphs.getDimension(
    st::Union{<:AbstractManifold, <:StateType}
)
    return manifold_dimension(getManifold(st))
end

function DistributedFactorGraphs.getDimension(
    hode::HomotopyDensity
)
    return getDimension(getStateKind(hode))
end


# FIXME, see near duplicate signature below -- must consolidate
function HomotopyDensity(
    bel::HD,
    partial_::L;
    observability::AbstractVector{<:Real} = bel.observability,
) where {HD <: HomotopyDensity, L <: Union{Nothing, <:AbstractVector{<:Integer}, <:Tuple}}
    #
    _workaround(::HomotopyDensityLive) = HomotopyDensityLive
    _workaround(::HomotopyDensityDFG) = HomotopyDensityDFG

    N = Npts(bel)
    partial = _tuple(partial_)
    mani = getManifold(bel)
    partl = _intersect(getPartial(bel), partial)
    M_, reprl, partl_cb = getManifoldPartial(
        mani, 
        partl, 
        bel.points[1],
    )
    if length(partl) != manifold_dimension(mani)
        # update representation kind to have correct partials
        _partialrepr(hr::HomotopyRepr) = HomotopyRepr(hr; partial = partl)
        reprkind = _partialrepr(bel.reprkind)
        # update majors to have correct partials
        for i in 1:length(bel.principal_elements)
            if isassigned(bel.principal_elements, i)
                nkn = _intersectpartials(mani, getKernelTree(bel, i), partial, partl_cb)
                bel.principal_elements[i] = mean(nkn)
                bel.principal_forms[i] = cov(nkn)
            end
        end

        # update minors detail to have correct partials
        nzs, _ = SparseArrays.findnz(bel.trailing_forms)
        for i in nzs
            cv = bel.trailing_forms[i]
            dummy = bel.points[1]
            cg = ConcentratedGaussianKernel(dummy, cv)
            cg_ = _intersectpartials(mani, cg, partial, partl_cb)
            cv_ = cov(cg_)
            bel.trailing_forms[i] = SMatrix{size(cv_)..., Float64}(cv_)
        end
        # update density to have correct partials
        _HD = _workaround(bel) # FIXME remove after partial types are stable - i.e. drop Nothing vs Tuple
        bel_ = _HD(;
            reprkind,
            observability,
            points = bel.points,
            weights = bel.weights,
            structure = bel.structure,
            principal_coeffs = bel.principal_coeffs,
            principal_elements = bel.principal_elements,
            principal_forms = bel.principal_forms,
            trailing_forms = bel.trailing_forms,
        )

        # call the constructor direct
        return bel_
    else
        # full manifold, i.e. partial=nothing
        return bel
    end
end


function HomotopyDensity_legacy(
    kind::Union{<:AbstractManifold, <:StateType},
    pts::AbstractVector;
    partial = nothing,
    bw = nothing, # diagm(ones(manifold_dimension(getManifold(kind)))),
    newbw::Bool = true,
    algo = Optim.NelderMead(),
    observability::AbstractVector{<:Real} = zeros(manifold_dimension(getManifold(kind))),
    weights::AbstractVector{<:Real} = ones(length(pts)) .* (1 / length(pts)),
    kw...
)
    __bw = if isnothing(bw)
        diagm(ones(manifold_dimension(getManifold(kind))))
    else
        bw
    end
    _legacybw(::Nothing) = __bw
    _legacybw(s::AbstractMatrix) = any(size(s) .== 1) ? diagm(vec(s)) : s
    _legacybw(s::AbstractVector) = diagm(s)

    manifold = getManifold(kind)

    M_, reprl, partl_cb = getManifoldPartial(manifold, partial, pts[1])

    hode = ApproxManifoldProducts.buildTree_Manellic!(
        kind,
        pts;
        kernel_bw = _legacybw(bw),
        kernel = ConcentratedGaussianKernel,
        partial = _tuple(partial),
        partl_cb,
        observability,
        weights,
    )

    # mask bw for partially excluded dimensions -- assumed 1.0 from legacy but...
    __partialCovToDefault!(s) = _partialCovToDefault!(_makevec(partial), s)

    # Cost function to optimize
    # avoid rebuilding tree at each optim iteration!!!
    _cost(σ::Real) =           entropy(hode,       [σ^2;;]                 )
    _cost(σ::AbstractVector) = entropy(hode, diagm(__partialCovToDefault!(σ .^ 2)))
    _cost(σ::AbstractMatrix) = entropy(hode,       __partialCovToDefault!(σ ^ 2)  )

    _bw(v::AbstractVector) = __partialCovToDefault!(_forcemutable(v))
    _bw(m::AbstractMatrix) = _bw(diag(_forcemutable(m)))

    # optimize for best LOOCV bandwidth
    # FIXME switch to RLM (or other Manopt) techinque instead 
    # set lower and upper bounds for Golden section optimization
    best_cov = if newbw && 1 === manifold_dimension(manifold)
        lcov, ucov = getBandwidthSearchBounds(hode)
        res =
            Optim.optimize((s) -> _cost([s;]), lcov[1], ucov[1], Optim.GoldenSection())
        [Optim.minimizer(res);;]
    elseif newbw
        bw0 = isnothing(bw) ? getBW(hode, false)[1] : bw # TODO consolidate with legacybw earlier
        res = Optim.optimize(
            _cost,
            _bw(bw0), # FIXME Optim API issue, if using bw::matrix then steps not PDMat (NelderMead) 
            algo,
        )
        diagm(abs.(Optim.minimizer(res)))
    else
        _legacybw(__bw)
    end
    __partialCovToDefault!(best_cov)

    # return tree with correct bandwidth
    return if newbw
        updateBandwidths(hode, best_cov; partl_cb)
    else
        hode
    end
end


