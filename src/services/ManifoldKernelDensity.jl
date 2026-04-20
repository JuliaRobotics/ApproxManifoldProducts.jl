

## ==========================================================================================
## helper functions to contruct MKD objects
## ==========================================================================================

getPointRepr(x::HomotopyDensity) = eltype(x.data) # TODO use HomotopyDensity{T} style instead
getManifold(x::HomotopyDensity) = x.manifold


Ndim(x::ManifoldKernelDensity, w...; kw...) = Ndim(x.belief, w...; kw...)
Npts(x::ManifoldKernelDensity, w...; kw...) = Npts(x.belief, w...; kw...)

getWeights(x::ManifoldKernelDensity, w...; kw...) = getWeights(x.belief, w...; kw...)

# getKDERange(x::ManifoldKernelDensity, w...; kw...) = getKDERange(x.belief, w...; kw...)
# function getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...; kw...)
#     return getKDERange((s -> s.belief).(x), w...; kw...)
# end
# getKDEMax(x::ManifoldKernelDensity, w...; kw...) = getKDEMax(x.belief, w...; kw...)
# getKDEMean(x::ManifoldKernelDensity, w...; kw...) = getKDEMean(x.belief, w...; kw...)
# getKDEfit(x::ManifoldKernelDensity, w...; kw...) = getKDEfit(x.belief, w...; kw...)

kld(x::ManifoldKernelDensity, w...; kw...) = kld(x.belief, w...; kw...)
minkld(x::ManifoldKernelDensity, w...; kw...) = minkld(x.belief, w...; kw...)

(x::ManifoldKernelDensity)(w...; kw...) = x.belief(w...; kw...)

getPointRepr(x::ManifoldKernelDensity) = getPointRepr(x.belief)
getManifold(x::ManifoldKernelDensity) = getManifold(x.belief)

function ManifoldKernelDensity(
    bel::B,
    ::Nothing = nothing;
    partl_cb::Nothing = nothing,
) where {B <: HomotopyDensity}
    return ManifoldKernelDensity{B, Nothing}(bel, nothing)
end


function ManifoldKernelDensity(
    bel::B,
    partial_::L;
    # u0::P = zeros(manifold_dimension(mani));
    infoPerCoord::AbstractVector{<:Real} = ones(getNumberCoords(getManifold(bel), bel.data[1])),
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {B <: HomotopyDensity, L <: AbstractVector{<:Integer}}
    #
    if isnothing(partl_cb)
        @warn "WIP partl_cb on MKD constructor helper" maxlog=10
    end
    partial = _tuple(partial_)
    mani = getManifold(bel)
    if length(partial) != manifold_dimension(mani)
        # TODO, assuming there are tree and leaf nodes at [1]...
        # @show getKernelTree(bel, 1)

        _tkT() = _intersectpartials(mani, getKernelTree(bel, 1), partial) |> typeof
        _lkT() = _intersectpartials(mani, getKernelLeaf(bel, 1), partial) |> typeof
        tree_kernels  = SizedVector{length(bel.tree_kernels), _tkT()}(undef)
        leaf_kernels  = SizedVector{length(bel.leaf_kernels), _lkT()}(undef)
        tkm = (s->isassigned(bel.tree_kernels, s)).(1:length(bel.tree_kernels))
        lkm = (s->isassigned(bel.leaf_kernels, s)).(1:length(bel.leaf_kernels))
        tree_kernels_ = view(tree_kernels, tkm)
        leaf_kernels_ = view(leaf_kernels, lkm)
        tree_kernels_ .= _intersectpartials.(Ref(mani), view(bel.tree_kernels, tkm), Ref(partial), partl_cb)
        leaf_kernels_ .= _intersectpartials.(Ref(mani), view(bel.leaf_kernels, lkm), Ref(partial), partl_cb)
        # TODO update belief to have correct partials
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
        # TODO remove _makevec on partials, here and everywhere really.
        return ManifoldKernelDensity{typeof(bel_), L}(bel_, _makevec(partial))
    else
        # full manifold, therefore equivalent to L::Nothing
        return ManifoldKernelDensity(bel, nothing)
    end
end

function ManifoldKernelDensity(
    bel::B,
    pl_mask::Union{<:BitVector, <:AbstractVector{<:Bool}},
) where {B <: HomotopyDensity}
    @warn "This constructor is not recommended, as partials have changed somewhat -- possibly erroneous code here..." maxlog=100
    return ManifoldKernelDensity(
        bel,
        (1:manifold_dimension(getManifold(bel)))[pl_mask], # TODO use tuple instead
    )
end


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
    return ManifoldKernelDensity(bel, partial)
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
    mkd::ManifoldKernelDensity{B, Nothing},
    field::Function,
    _aspartial::Bool = true,
) where {B}
    return field(mkd)
end

function _getFieldPartials(
    mkd::ManifoldKernelDensity{B, <:AbstractVector},
    field::Function,
    aspartial::Bool = true,
) where {B}
    _length(x::AbstractMatrix) = length(diag(x))
    _length(x::AbstractVector) = length(x)
    val = field(mkd)
    if aspartial && (_length(val) == length(mkd._partial))
        return val
    elseif !aspartial && (_length(val) == length(mkd._partial))
        val_ = zeros(manifold_dimension(getManifold(mkd)))
        val_[mkd._partial] .= val
        return val_
    elseif aspartial && (_length(val) == manifold_dimension(getManifold(mkd)))
        return val[mkd._partial]
    elseif !aspartial && (_length(val) == manifold_dimension(getManifold(mkd)))
        return val
    else
        error(
            "unknown size MKD.$(field) with partial length=$(length(mkd._partial)) vs length=$(_length(val)) --- and value=$val",
        )
    end
end

function getInfoPerCoord(mkd::ManifoldKernelDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> x.belief.infoPerCoord, aspartial)
end

function getBandwidth(mkd::ManifoldKernelDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> getBW(x)[1], aspartial)
end


"""
    $SIGNATURES

Return true if this ManifoldKernelDensity is a partial.
"""
isPartial(mkd::ManifoldKernelDensity{B, L}) where {B, L} = true
isPartial(mkd::ManifoldKernelDensity{B, Nothing}) where {B} = false

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
    x::ManifoldKernelDensity{B, Nothing},
    ::Bool = true; # aspartial unused
    permute::Bool = true,
) where {B}
    return getPoints(x.belief; permute)
end


function getPoints(
    x::ManifoldKernelDensity{B, L},
    aspartial::Bool = true;
    permute::Bool = true,
) where {B, L <: AbstractVector{Int}}
    #
    pts = getPoints(x.belief; permute)

    if (L === nothing) && !aspartial
        error("MKD getPoints aspartial=true but MKD is not partial")
        return pts
    end

    Mp, Rp, lkup = getManifoldPartial(getManifold(x), x._partial, pts[1])

    vecP = Vector{typeof(Rp)}(undef, length(pts))
    for (j,pt) in enumerate(pts)
        vecP[j] =  lkup(pt)
    end
    return vecP
end


function getBW(
    x::ManifoldKernelDensity{B, L},
    asPartial::Bool = true;
    kw...,
) where {B, L}
    bws = getBW(x.belief; kw...)
    if L !== Nothing && asPartial
        return (bw->view(bw, x._partial)).(bws)
    end
    return bws
end

# TODO check that partials / marginals are sampled correctly
function sample(x::ManifoldKernelDensity{B, L}, N::Integer = 1) where {B, L}
    # get legacy matrix of coordinates and selected labels
    belief = x.belief
    coords, lbls = sample(belief, N)
    # pack samples into vector of point type P
    vecP = Vector{eltype(belief.data)}(undef, N)
    for j = 1:N
        vecP[j] = makePointFromCoords(getManifold(x), view(coords, :, j), belief.data[1])
    end

    return vecP, lbls
end

Random.rand(mkd::ManifoldKernelDensity, N::Integer) = sample(mkd, N)[1]

# rand(mkd::ManifoldKernelDensity) = sample(mkd, 1)[1][1]
Distributions.variate_form(mkd::ManifoldKernelDensity) = Ndim(mkd) == 1 ? Univariate : Multivariate
function Random.rand(mkd::ManifoldKernelDensity)
    return _rand(Distributions.variate_form(mkd), mkd)
end
_rand(::Type{Univariate}, mkd::ManifoldKernelDensity) = sample(mkd.belief, 1)[1][:][]
_rand(::Type{Multivariate}, mkd::ManifoldKernelDensity) = sample(mkd.belief, 1)[1][:]

function resample(x::ManifoldKernelDensity, N::Int)
    pts = if N < Npts(x)
        # get points with non-partial coord dims so that new MKD can be built
        shuffle(getPoints(x, false))[1:N]
    else
        _pts, = sample(x, N)
        _pts
    end
    return ManifoldKernelDensity(
        getManifold(x),
        pts;
        partial = x._partial,
        infoPerCoord = x.belief.infoPerCoord,
    )
end

function Base.show(io::IO, mkd::ManifoldKernelDensity{B, L}) where {B, L}
    _round(s::AbstractArray; kw...) = round.(s[:]; kw...)
    _round(s::AbstractVector{<:AbstractMatrix}; kw...) = round.(s[1][:]; kw...)

    printstyled(io, "ManifoldKernelDensity{"; bold = true, color = :blue)
    println(io)
    # FIXME restore after HomotopyDensity refactor
    # printstyled(io, "    M"; bold = true, color = :magenta)
    # print(io, " = ", M, ",")
    # println(io)
    printstyled(io, "    B"; bold = true, color = :magenta)
    print(io, " = ", B, ",")
    println(io)
    printstyled(io, "    L"; bold = true, color = :magenta)
    print(io, " = ", L, ",")
    println(io)
    println(io, " }(")
    println(io, "  Npts:  ", Npts(mkd.belief))
    print(io, "  dims:  ", Ndim(mkd.belief))
    printstyled(io, isPartial(mkd) ? "* --> $(length(mkd._partial))" : ""; bold = true)
    println(io)
    println(io, "  prtl:   ", mkd._partial)
    bw = (getBW(mkd.belief).^2)[:, 1]
    pvec = isPartial(mkd) ? mkd._partial : collect(1:length(bw))
    println(io, "  bws:   ", getBandwidth(mkd, true) |> x -> _round(x; digits = 4)) # .|> x->round(x,digits=4))
    println(io, "  ipc:   ", getInfoPerCoord(mkd, true) .|> x -> round(x; digits = 4))
    print(io, "   mean: ")
    try
        mn = mean(mkd)
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

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)
function Base.show(io::IO, ::MIME"application/juno.inline", mkd::ManifoldKernelDensity)
    return show(io, mkd)
end

# override
function marginal(
    x::ManifoldKernelDensity{B},
    dims::AbstractVector{<:Integer},
) where {B}
    #
    ldims::Vector{Int} = _makevec(_intersect(x._partial, collect(dims)))
    return ManifoldKernelDensity(x.belief, ldims)
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
    mkd::ManifoldKernelDensity,
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



function Statistics.mean(mkd::ManifoldKernelDensity, aspartial::Bool = true; kwargs...)
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
calcMean(mkd::ManifoldKernelDensity, aspartial::Bool = true) = mean(mkd, aspartial)

function Statistics.std(mkd::ManifoldKernelDensity, aspartial::Bool = true; kwargs...)
    return std(_getManifoldFullOrPart(mkd, aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.var(mkd::ManifoldKernelDensity, aspartial::Bool = true; kwargs...)
    return var(_getManifoldFullOrPart(mkd, aspartial), getPoints(mkd, aspartial); kwargs...)
end
function Statistics.cov(
    mkd::ManifoldKernelDensity,
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
