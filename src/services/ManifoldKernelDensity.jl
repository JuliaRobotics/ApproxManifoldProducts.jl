

## ==========================================================================================
## helper functions to contruct MKD objects
## ==========================================================================================

getPointRepr(x::HomotopyDensity) = eltype(x.data) # TODO use HomotopyDensity{T} style instead
function getManifold(x::HomotopyDensity, aspartial::Bool = false)
    return if !aspartial
        x.manifold
    else
        M_, _, _ = getManifoldPartial(x.manifold, getPartial(x), x.data[1])
        M_
    end
end

getPartial(::HomotopyDensity{partial}) where {partial} = partial


# getKDERange(x::ManifoldKernelDensity, w...; kw...) = getKDERange(x.shim, w...; kw...)
# function getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...; kw...)
#     return getKDERange(x, w...; kw...)
# end
# getKDEMax(x::ManifoldKernelDensity, w...; kw...) = getKDEMax(x.shim, w...; kw...)
# getKDEMean(x::ManifoldKernelDensity, w...; kw...) = getKDEMean(x.shim, w...; kw...)
# getKDEfit(x::ManifoldKernelDensity, w...; kw...) = getKDEfit(x.shim, w...; kw...)

# kld(x::ManifoldKernelDensity, w...; kw...) = kld(x.shim, w...; kw...)
# minkld(x::ManifoldKernelDensity, w...; kw...) = minkld(x.shim, w...; kw...)



function HomotopyDensity(
    bel::HomotopyDensity,
    partial_::L;
    infoPerCoord::AbstractVector{<:Real} = bel.infoPerCoord,
) where {L <: Union{<:AbstractVector{<:Integer}, <:Tuple}}
    #
    partial = _tuple(partial_)
    mani = getManifold(bel)
    partl = _intersect(getPartial(bel), partial)
    M_, reprl, partl_cb = getManifoldPartial(
        mani, 
        partl, 
        bel.data[1],
    )
    if length(partl) != manifold_dimension(mani)
        # assuming there are tree and leaf nodes at [1]...
        _tkT() = _intersectpartials(mani, getKernelTree(bel, 1), partial) |> typeof
        _lkT() = _intersectpartials(mani, getKernelLeaf(bel, 1), partial) |> typeof
        tree_kernels  = SizedVector{length(bel.tree_kernels), _tkT()}(undef)
        leaf_kernels  = SizedVector{length(bel.leaf_kernels), _lkT()}(undef)
        tkm = (s->isassigned(bel.tree_kernels, s)).(1:length(bel.tree_kernels))
        lkm = (s->isassigned(bel.leaf_kernels, s)).(1:length(bel.leaf_kernels))
        tree_kernels_ = view(tree_kernels, tkm)
        leaf_kernels_ = view(leaf_kernels, lkm)
        tree_kernels_ .= (s->_intersectpartials(mani, s, partial, partl_cb)).(view(bel.tree_kernels, tkm))
        leaf_kernels_ .= (s->_intersectpartials(mani, s, partial, partl_cb)).(view(bel.leaf_kernels, lkm))
        # update belief to have correct partials
        bel_ = HomotopyDensity{
            _getprl(eltype(tree_kernels)),
        }(;
            manifold = getManifold(bel),
            data = bel.data,
            weights = bel.weights,
            permute = bel.permute,
            leaf_kernels,
            tree_kernels,
            infoPerCoord,
            segments = bel.segments,
            _workaround_isdef_leafkernel = bel._workaround_isdef_leafkernel,
            _workaround_isdef_treekernel = bel._workaround_isdef_treekernel,
        )

        # call the constructor direct
        return bel_
    else
        # full manifold, i.e. partial=nothing
        return bel
    end
end



# override
marginal(
    hode::HomotopyDensity,
    partl::AbstractVector{<:Integer},
) = HomotopyDensity(hode, partl)



# previously manikde!_manellic
function manikde!(
    M::AbstractManifold,
    pts::AbstractVector;
    bw = diagm(ones(manifold_dimension(M))),
    algo = Optim.NelderMead(),
    partial::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    kw...
)
    #
    M_, reprl, partl_cb = getManifoldPartial(M, partial, pts[1])

    mtree = ApproxManifoldProducts.buildTree_Manellic!(
        M,
        pts;
        kernel_bw = bw,
        kernel = ConcentratedGaussianKernel,
        partial,
        partl_cb,
    )

    # mask bw for partially excluded dimensions -- assumed 1.0 from legacy but...
    __partialCovToDefault!(s) = _partialCovToDefault!(partial, s)

    # Cost function to optimize
    # avoid rebuilding tree at each optim iteration!!!
    _cost(σ::Real) =           entropy(mtree,       [σ^2;;]                 )
    _cost(σ::AbstractVector) = entropy(mtree, diagm(__partialCovToDefault!(σ .^ 2)))
    _cost(σ::AbstractMatrix) = entropy(mtree,       __partialCovToDefault!(σ ^ 2)  )

    _bw(v::AbstractVector) = __partialCovToDefault!(v)
    _bw(m::AbstractMatrix) = _bw(diag(m))

    # optimize for best LOOCV bandwidth
    # FIXME switch to RLM (or other Manopt) techinque instead 
    # set lower and upper bounds for Golden section optimization
    best_cov = if 1 === manifold_dimension(M)
        lcov, ucov = getBandwidthSearchBounds(mtree)
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

    bel = updateBandwidths(mtree, best_cov; partl_cb)
    # return tree with correct bandwidth
    return bel
end



## ==========================================================================================
## a few utilities
## ==========================================================================================



# partial (i.e. active) coordinate dimensions are left unchanged, while inactive 
# dimensions are set to default values (1.0 for variances, 0.0 for covariances)
_partialCovToDefault!(::Nothing, s) = s
function _partialCovToDefault!(p::Union{<:Tuple, <:AbstractVector{<:Integer}}, v::AbstractVector)
    mask = ones(Int, length(v)) .== 1
    mask[p] .= false
    v[mask] .= 1.0
    return v
end
function _partialCovToDefault!(p::Union{<:Tuple, <:AbstractVector{<:Integer}}, m::AbstractMatrix)
    for i in axes(m, 1)
        for j in axes(m, 2)
            if !(i in p) || !(j in p)
                # default values for inactive elements of covariance matrix
                m[i,j] = i == j ? Inf : 0.0
            end
            # else leave row and column unchanged
        end
    end
    return m
end


function _getFieldPartials(
    mkd::HomotopyDensity{partial},
    field::Function,
    aspartial::Bool = true,
) where {partial}
    if isnothing(partial)
        return field(mkd)
    end
    _length(x::AbstractMatrix) = length(diag(x))
    _length(x::AbstractVector) = length(x)
    val = field(mkd)
    if aspartial && (_length(val) == length(getPartial(mkd)))
        return val
    elseif !aspartial && (_length(val) == length(getPartial(mkd)))
        val_ = zeros(manifold_dimension(getManifold(mkd)))
        val_[getPartial(mkd)] .= val
        return val_
    elseif aspartial && (_length(val) == manifold_dimension(getManifold(mkd)))
        return val[_makevec(getPartial(mkd))]
    elseif !aspartial && (_length(val) == manifold_dimension(getManifold(mkd)))
        return val
    else
        error(
            "unknown size MKD.$(field) with partial length=$(length(getPartial(mkd))) vs length=$(_length(val)) --- and value=$val",
        )
    end
end

function getInfoPerCoord(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> x.infoPerCoord, aspartial)
end

function getBandwidth(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> getBW(x)[1], aspartial)
end


"""
    $SIGNATURES

Return true if this HomotopyDensity is a partial.
"""
isPartial(::HomotopyDensity{partl}) where partl = !isnothing(partl)


# TODO check that partials / marginals are sampled correctly
function sample(belief::HomotopyDensity, N::Integer = 1)
    # get legacy matrix of coordinates and selected labels
    coords, lbls = sample(belief, N)
    # pack samples into vector of point type P
    vecP = Vector{eltype(belief.data)}(undef, N)
    for j = 1:N
        vecP[j] = makePointFromCoords(getManifold(belief), view(coords, :, j), belief.data[1])
    end

    return vecP, lbls
end

Random.rand(hode::HomotopyDensity, N::Integer) = sample(hode, N)[1]

# rand(hode::HomotopyDensity) = sample(hode, 1)[1][1]
Distributions.variate_form(hode::HomotopyDensity) = Ndim(hode) == 1 ? Univariate : Multivariate
function Random.rand(hode::HomotopyDensity)
    return _rand(Distributions.variate_form(hode), hode)
end
_rand(::Type{Univariate}, hode::HomotopyDensity) = sample(hode, 1)[1][:][]
_rand(::Type{Multivariate}, hode::HomotopyDensity) = sample(hode, 1)[1][:]

function resample(x::HomotopyDensity, N::Int)
    pts = if N < Npts(x)
        # get points with non-partial coord dims so that new MKD can be built
        shuffle(getPoints(x, false))[1:N]
    else
        _pts, = sample(x, N)
        _pts
    end
    return HomotopyDensity(
        getManifold(x),
        pts;
        partial = getPartial(x),
        infoPerCoord = x.infoPerCoord,
    )
end




"""
    $SIGNATURES

If a marginal (statistics) of a probability reduce the dimensions (i.e. casts a shadow, or projects); then an 
antimarginal aims to increase dimension of the probability within reason.

Notes
- Marginalization is a integration of for higher dimension to lower dimension, so antimarginal likely involve 
  differentiation (anti-integral) instead.
- In manifold language, this is an embedding into a higher dimensional space.
- See structure from motion in machine vision, or stereo disparity for building depth clouds from 2D images.
- Imagine combining three different partial embedding A=[1, nan, nan, 1.4], B=[nan,2.1,nan,4], C=[nan, nan, 3, nan]
  which should equal A+B+C = [notnan, notnan, notnan, notnan]
"""
function antimarginal(
    newM::AbstractManifold,
    u0,
    mkd::HomotopyDensity,
    newpartial::AbstractVector{<:Integer},
)
    #

    # convert to antimarginal by copying user provided example point for bigger manifold
    pts = getPoints(mkd, false)
    # new coord partials must be placed into a full dimension point, thats why we use u0
    nPts = Vector{typeof(u0)}(undef, length(pts))
    for i in eachindex(pts)
        setPointPartial!(newM, nPts, getManifold(mkd), pts, newpartial, i)
    end

    # also update metadata elements
    finalpartial =
        !isPartial(mkd) ? newpartial : error("not built yet, to shift incoming partial")
    bw = zeros(manifold_dimension(newM))
    bw[finalpartial] .= getBW(mkd)[:, 1]
    ipc = zeros(manifold_dimension(newM))
    ipc[finalpartial] .= getInfoPerCoord(mkd, true)

    return manikde!(newM, nPts, u0; bw, partial = finalpartial, infoPerCoord = ipc)
end



function Statistics.mean(mkd::HomotopyDensity, aspartial::Bool = true; kwargs...)
    return mean(
        _getManifoldFullOrPart(mkd, aspartial),
        getPoints(mkd, aspartial),
        GeodesicInterpolation();
        kwargs...,
    )
end
"""
    $SIGNATURES

Alias for overloaded `Statistics.mean`.
"""
calcMean(mkd::HomotopyDensity, aspartial::Bool = true) = mean(mkd, aspartial)

function Statistics.std(mkd::HomotopyDensity, aspartial::Bool = true; kwargs...)
    return std(_getManifoldFullOrPart(mkd, aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.var(mkd::HomotopyDensity, aspartial::Bool = true; kwargs...)
    return var(_getManifoldFullOrPart(mkd, aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.cov(
    mkd::HomotopyDensity,
    aspartial::Bool = true;
    basis::ManifoldsBase.AbstractBasis = DefaultOrthogonalBasis(),
    kwargs...,
)
    return cov(
        _getManifoldFullOrPart(mkd, aspartial),
        getPoints(mkd, aspartial);
        basis,
        kwargs...,
    )
end


#
