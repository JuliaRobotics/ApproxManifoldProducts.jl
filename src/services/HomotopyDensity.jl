

## ==========================================================================================
## HomotopyDensity constructorhelper functions
## ==========================================================================================

# overload Base.convert for easy conversion between live and hold representations
convert(::Type{<:HomotopyDensityLive}, src::HomotopyDensityHold) = HomotopyDensityLive(src)
convert(::Type{<:HomotopyDensityHold}, src::HomotopyDensityLive) = HomotopyDensityHold(src)


HomotopyDensityLive(hode::HomotopyDensityHold) = HomotopyDensityLive(
  hode.reprkind,
  hode.observability,
  hode.points,
  hode.weights,
  hode.majors_coeff,
  hode.majors_element,
  hode.majors_detail,
  hode.minors_detail,
  hode.structure
)
HomotopyDensityHold(hode::HomotopyDensityLive) = HomotopyDensityHold(
  hode.reprkind,
  hode.observability,
  hode.points,
  hode.weights,
  hode.majors_coeff,
  hode.majors_element,
  hode.majors_detail,
  hode.minors_detail,
  hode.structure
)

# FIXME, see near duplicate signature below -- must consolidate
function HomotopyDensity(
    bel::HD,
    partial_::L;
    observability::AbstractVector{<:Real} = bel.observability,
) where {HD <: HomotopyDensity, L <: Union{Nothing, <:AbstractVector{<:Integer}, <:Tuple}}
    #
    _workaround(::HomotopyDensityLive) = HomotopyDensityLive
    _workaround(::HomotopyDensityHold) = HomotopyDensityHold

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
        _partialrepr(::HomotopyRepresentation{M, L, K, D}) where {M, L, K, D} = HomotopyRepresentation{M, typeof(partl), K, D}(mani, partl)
        reprkind = _partialrepr(bel.reprkind)
        # update majors to have correct partials
        for i in 1:length(bel.majors_element)
            if isassigned(bel.majors_element, i)
                nkn = _intersectpartials(mani, getKernelTree(bel, i), partial, partl_cb)
                bel.majors_element[i] = mean(nkn)
                bel.majors_detail[i] = cov(nkn)
            end
        end

        # update minors detail to have correct partials
        nzs, _ = SparseArrays.findnz(bel.minors_detail)
        for i in nzs
            cv = bel.minors_detail[i]
            dummy = bel.points[1]
            cg = ConcentratedGaussianKernel(dummy, cv)
            cg_ = _intersectpartials(mani, cg, partial, partl_cb)
            cv_ = cov(cg_)
            bel.minors_detail[i] = SMatrix{size(cv_)..., Float64}(cv_)
        end
        # update density to have correct partials
        _HD = _workaround(bel) # FIXME remove after partial types are stable - i.e. drop Nothing vs Tuple
        bel_ = _HD(;
            reprkind,
            observability,
            points = bel.points,
            weights = bel.weights,
            structure = bel.structure,
            majors_coeff = bel.majors_coeff,
            majors_element = bel.majors_element,
            majors_detail = bel.majors_detail,
            minors_detail = bel.minors_detail,
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
    bw = diagm(ones(manifold_dimension(getManifold(kind)))),
    algo = Optim.NelderMead(),
    kw...
)
    #
    manifold = getManifold(kind)

    M_, reprl, partl_cb = getManifoldPartial(manifold, partial, pts[1])

    hode = ApproxManifoldProducts.buildTree_Manellic!(
        manifold,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
        partial = _tuple(partial),
        partl_cb,
    )

    # mask bw for partially excluded dimensions -- assumed 1.0 from legacy but...
    __partialCovToDefault!(s) = _partialCovToDefault!(_makevec(partial), s)

    # Cost function to optimize
    # avoid rebuilding tree at each optim iteration!!!
    _cost(σ::Real) =           entropy(hode,       [σ^2;;]                 )
    _cost(σ::AbstractVector) = entropy(hode, diagm(__partialCovToDefault!(σ .^ 2)))
    _cost(σ::AbstractMatrix) = entropy(hode,       __partialCovToDefault!(σ ^ 2)  )

    _bw(v::AbstractVector) = __partialCovToDefault!(v)
    _bw(m::AbstractMatrix) = _bw(diag(m))

    # optimize for best LOOCV bandwidth
    # FIXME switch to RLM (or other Manopt) techinque instead 
    # set lower and upper bounds for Golden section optimization
    best_cov = if 1 === manifold_dimension(manifold)
        lcov, ucov = getBandwidthSearchBounds(hode)
        res =
            Optim.optimize((s) -> _cost([s;]), lcov[1], ucov[1], Optim.GoldenSection())
        [Optim.minimizer(res);;]
    else
        res = Optim.optimize(
            _cost,
            _bw(bw), # FIXME Optim API issue, if using bw::matrix then steps not PDMat (NelderMead) 
            algo,
        )
        diagm(abs.(Optim.minimizer(res)))
    end
    __partialCovToDefault!(best_cov)

    belief = updateBandwidths(hode, best_cov; partl_cb)
    # return tree with correct bandwidth
    return belief
end


