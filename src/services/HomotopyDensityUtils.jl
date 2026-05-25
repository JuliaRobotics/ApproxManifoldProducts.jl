


## ==========================================================================================
## UX/DX Convenience functions
## ==========================================================================================


# FIXME, heavy legacy -- update this to a prettier show of modern HomotopyDensity
function Base.show(io::IO, hode::HomotopyDensity)
    _getP(::HomotopyDensityDFG{H, P}) where {H,P} = P
    _getP(::HomotopyDensityLive{H, P}) where {H,P} = P
    N = Npts(hode)
    printstyled(io, "HomotopyDensity{"; bold = true, color = :blue)
    println(io)
    printstyled(io, "    partial"; bold = true, color = :magenta)
    print(io, " = ", getPartial(hode), ",")
    println(io)
    printstyled(io, "    M"; bold = true, color = :magenta)
    print(io, " = ", typeof(getManifold(hode.reprkind)), ",")
    println(io)
    printstyled(io, "  P  = ", _getP(hode); color = :magenta)
    println(io)
    printstyled(io, "  N  = ", N; color = :magenta)
    println(io)
    # printstyled(io, "  HL = ", HL; color = :magenta)
    # println(io)
    # printstyled(io, "  HT = ", HT, color = :magenta)
    # println(io)
    printstyled(io, "}"; bold = true, color = :blue)
    println(io, "(")
    @assert Npts(hode) == length(hode.points) "show(::HomotopyDensity,) noticed a data size issue, expecting N$(Npts(hode)) == length(.points)$(length(hode.points))"
    if 0 < Npts(hode)
        print(io, "  .points      :  ")
        0 < N ? println(io, hode.points[1], " ... ", hode.points[end]) : println(io, "[]")
        print(io, "  .weights     :  ")
        0 < length(hode.weights) ? println(io, hode.weights[1], " ... ", hode.weights[end]) : println(io, "[]")
        # printstyled(io, "     (uniwt)  :   ", uniWT(hode); color = :light_black)
        println(io)
        print(io, "  .structure[1][-]:  ")
        if 0 < length(hode.structure.nzind) && 0 < length(hode.structure.nzval[1])
            printstyled(io, hode.structure.nzval[1][1], " ... ", hode.structure.nzval[1][end]; color = :light_black)
            println(io)
        end
        print(io, "  .tkernels[") # " __see below__"; color=:light_black)
        if 0 < Npts(hode)
            # printstyled(io, "  .tkernels[1] = "; color=:light_black)
            print(io, "1]:  ")
            printstyled(io, "::HT "; color = :magenta)
            if isassigned(hode, 1)
                try
                    printstyled(io, getKernelTree(hode, 1); color = :light_black)
                catch e
                    if e isa PosDefException
                        printstyled(io, "_PosDefEx_"; color =:red)
                    else
                        printstyled(io, "_unable_"; color = :red)
                    end
                end
            else
                printstyled(io, "undef"; color = :red)
            end
            println(io)
            # print(io, "  ...,")
        else
            print(io, "]:   ")
            printstyled(io, "::HT "; color = :magenta)
            println(io)
        end
        printstyled(
            io,
            "     (trunc)  :   ",
            getTopologyKind(hode.reprkind);
            color = :light_black,
        )
        println(io)
        printstyled(io, "     (blncd)  :   ", "true : _wip_"; color = :light_black)
        println(io)
        print(io, "  .lkernels[")
        # if 0 < Npts(hode)
        #     print(io, "1]:  ")
        #     # printstyled(io, "  .tkernels[1] = "; color=:light_black)
        #     if isassigned(hode, 1+N)
        #         lk = getKernelLeaf(hode, 1)
        #         printstyled(io, lk; color = :light_black)
        #     else
        #         printstyled(io, "undef"; color = :red)
        #         println(io)
        #     end
        #     # print(io, "  ...,")
        #     if 1 < Npts(hode)
        #         printstyled(io, "         [end]:  "; color = :light_black)
        #         if isassigned(hode.leaf_kernels, length(hode.leaf_kernels))
        #             lk = getKernelLeaf(hode, Npts(hode))
        #             printstyled(io, lk; color = :light_black)
        #         else
        #             printstyled(io, "undef"; color = :red)
        #             println(io)
        #         end
        #     end
        # else
        #     print(io, "]:   ")
        #     println(io)
        # end
        # # printstyled(io, "{1..$N}"; color=:light_black)
        # # println(io)
        # uBW = uniBW(hode)
        # printstyled(io, "     (unibw)  :   ", uBW; color = :light_black)
        # println(io)
        # if uBW
        #     printstyled(
        #         io,
        #         "         bw   :    ",
        #         round.((getBW(hode).^2)[1][:]'; digits = 3);
        #         color = :light_black,
        #     )
        #     println(io)
        # end
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
    println(io, "  ipc:   ", getObservability(hode, true) .|> x -> round(x; digits = 4))
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




## ==========================================================================================
## a few utilities
## ==========================================================================================

"""
    $SIGNATURES

Alias for overloaded `Statistics.mean`.
"""
calcMean(mkd::HomotopyDensity, aspartial::Bool = true) = mean(mkd, aspartial)


function Statistics.mean(mkd::HomotopyDensity, aspartial::Bool = true; kwargs...)
    # FIXME, should just pull tree node 1 mean
    return mean(
        _getManifoldFullOrPart(mkd, aspartial),
        getPoints(mkd, aspartial),
        GeodesicInterpolation();
        kwargs...,
    )
end
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


# getKDERange(x::ManifoldKernelDensity, w...; kw...) = getKDERange(x.shim, w...; kw...)
# function getKDERange(x::AbstractVector{<:ManifoldKernelDensity}, w...; kw...)
#     return getKDERange(x, w...; kw...)
# end
# getKDEMax(x::ManifoldKernelDensity, w...; kw...) = getKDEMax(x.shim, w...; kw...)
# getKDEMean(x::ManifoldKernelDensity, w...; kw...) = getKDEMean(x.shim, w...; kw...)
# getKDEfit(x::ManifoldKernelDensity, w...; kw...) = getKDEfit(x.shim, w...; kw...)

# kld(x::ManifoldKernelDensity, w...; kw...) = kld(x.shim, w...; kw...)
# minkld(x::ManifoldKernelDensity, w...; kw...) = minkld(x.shim, w...; kw...)


# number of data points (aka particles) in tree, i.e. N
Base.length(hode::HomotopyDensity) = Ndim(hode)
Npts(hode::HomotopyDensity) = length(hode.points)
Ndim(hode::HomotopyDensity) = manifold_dimension(getManifold(hode))

getWeights(mt::HomotopyDensity; permute::Bool = true) = permute ? view(mt.weights, mt.structure[1]) : mt.weights


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
    hode::HomotopyDensity,
    aspartial::Bool = true;
    permute::Bool = true,
)
    #
    partl = getPartial(hode)
    pts = permute ? view(hode.points, hode.structure[1]) : hode.points

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


"""
    getBW

Return bandwidth(s) of kernel(s) in homotopy density as variance.
"""
function getBW(
    hode::HomotopyDensity,
    aspartial::Bool = true,
)
    partl = getPartial(hode)
    # bws = (s->getBW(getKernelLeaf(hode, s))).(1:Npts(hode))
    # FIXME, Hack assuming parametric or nonparametric always Gaussian
    bws = if 1 == Npts(hode)
        hode.principal_forms
    elseif 1 < Npts(hode)
        hode.trailing_forms
    else
        error("This homotopy density has no principal or trailing covariance/bw.")
    end
    if isnothing(partl) && aspartial
        return (bw->_getpartial(partl, bw)).(bws)
    end
    return bws
end



# check for uniform weights
uniWT(mt::HomotopyDensity) = 1 === length(union(diff(getWeights(mt))))


# check for uniform bandwidths in kernels
function uniBW(hode::HomotopyDensity)
    N = Npts(hode)
    if !isassigned(hode, N+1)
        return false
    end
    # check equality on all bandwidths and return false if difference found
    lk = getKernelLeaf(hode, 1)
    bw = cov(lk)
    for i in 2:N
        lk = getKernelLeaf(hode, i)
        if !isapprox(bw, cov(lk))
            return false
        end
    end
    return true
end



function getObservability(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> x.observability, aspartial)
end

function getBandwidth(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> getBW(x)[1], aspartial)
end


function sample(
    hode::HomotopyDensity, 
    Npts::Integer=1
)
  _pointtype(::HomotopyDensityDFG{H, P}) where {H,P} = P
  _pointtype(::HomotopyDensityLive{H, P}) where {H,P} = P
  P = _pointtype(hode)
    # TODO, this is currently a bit of a hack to get samples out in the right format, needs refactor and cleanup
  _Compose(m::AbstractLieGroup, p, x) = LieGroups.compose(m, p, exp(m, hat(LieAlgebra(m), x, P)))
  _Compose(m::AbstractManifold, p, x) = Manifolds.compose(m, p, exp(m, p, hat(m, p, x)))
  manif = getManifold(hode)
  w = hode.weights
  ind = zeros(Int, Npts)
  c = Categorical(w)
  points = Vector{P}(undef, Npts)
  for i in 1:Npts
    lidx = rand(c)
    ind[i] = lidx
    ker = getKernelLeaf(hode, lidx) # only ConcentratedGaussian during Homotopy refac
    Xc = rand(ker.functional)
    p = mean(ker)
    points[i] = _Compose(manif, p, Xc)
  end
  return points, ind
end
# TODO check that partials / marginals are sampled correctly
# function sample(belief::HomotopyDensity, N::Integer = 1)
#     # get legacy matrix of coordinates and selected labels
#     coords, lbls = sample(belief, N)
#     # pack samples into vector of point type P
#     vecP = Vector{eltype(belief.points)}(undef, N)
#     for j = 1:N
#         vecP[j] = makePointFromCoords(getManifold(belief), view(coords, :, j), belief.points[1])
#     end

#     return vecP, lbls
# end

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
    return HomotopyDensity_legacy(
        getStateKind(x),
        pts;
        partial = getPartial(x),
        observability = x.observability,
    )
end


function updateBandwidths(
    hode::HD, 
    bws;
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {HD <: HomotopyDensity}
    #
    _getBW(s::Float64, ::Int) = [s;;]
    _getBW(s::AbstractVector{<:Real}, ::Int) = s
    _getBW(s::AbstractMatrix{<:Real}, ::Int) = s
    _getBW(s::AbstractVector{<:AbstractArray}, _i::Int) = s[_i]

    N = Npts(hode)

    (nzi,_) = SparseArrays.findnz(hode.trailing_forms)

    # leaf_kernels = Vector{HL}(undef, N)
    # for (i, lk) in enumerate(hode.leaf_kernels)
    for i in nzi
        nkl = ConcentratedGaussianKernel(getKernelLeaf(hode, i); Σ = _getBW(bws, i), partl_cb)
        # leaf_kernels[i] = nkl # updateKernelBW(lk, _getBW(bws, i))
        # new replacement field instead of .leaf_kernels
        # if i in nzi
            cv = cov(nkl)
            hode.trailing_forms[i] = _forcestatic(cv)
        # end
    end
    partial = getPartial(hode)
    reprkind = HomotopyRepr(
        hode.reprkind;
        partial,
    )
    # kind = getManifold(hode) 
    # reprkind = HomotopyRepr{
    #     BinaryTruncFixedDepth{3},
    #     ConcentratedGaussianKernel,
    #     typeof(kind),
    #     typeof(partial),
    # }(kind, partial)

    return HD(;
        reprkind,
        points = hode.points,
        weights = hode.weights,
        structure = hode.structure,
        principal_coeffs = hode.principal_coeffs,
        principal_elements = hode.principal_elements,
        principal_forms = hode.principal_forms,
        trailing_forms = hode.trailing_forms,
    )
end

"""
    $SIGNATURES
    
For Manellic tree parent kernels, what is the 'smallest' and 'biggest' covariance.

Notes:
- Thought about `det` for covariance volume but long access of pancake (smaller volume) is not minimum compared to circular covariance. 
"""
function getBandwidthSearchBounds(hode::HomotopyDensity)
    upper = cov(getKernelTree(hode, 1))
    
    lower_diag = diag(cov(getKernelTree(hode, 1)))
    for i in 2:(length(hode.principal_forms) - 1)
        if isassigned(hode.principal_forms, i)
            hdg = hcat(lower_diag, diag(cov(getKernelTree(hode, i))))
            lower_diag = minimum(hdg; dims = 2)
        end
        # lower_diag = minimum(hcat(lower_diag, diag(cov(getKernelTree(hode, i)))); dims = 2)
    end

    # floors make us feel safe, but hurt when faceplanting
    lower_diag = maximum(hcat(lower_diag, 1e-8 * ones(length(lower_diag))); dims = 2)[:]

    # FIXME override nans case -- was adding during refactor upgrade AMP v0.11
    lower_diag[isnan.(lower_diag)] .= 1e-8

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
    hode::HomotopyDensity,
    pt,
    LOO::Bool = false,
    force_kbw = nothing,
)
    partl = getPartial(hode)
    # # force function barrier, just to be sure dyndispatch is limited
    # _F() = getfield(ApproxManifoldProducts,HL.name.name)
    # _F_ = _F() 

    pts = getPoints(hode, false)
    w = getWeights(hode)

    manif = getManifold(hode)
    # isapprox uses partial version
    M_, reprl, cb = getManifoldPartial(manif, partl)
    sumval = 0.0
    # FIXME, brute force for loop
    for (i, t) in enumerate(pts)
        if !LOO || !isapprox(M_, cb(pt), cb(t))
            ekr = getKernelLeaf(hode, i)
            ekr = updateKernelBW(ekr, force_kbw)
            # remember special handling for partials via ekr itself
            oneval = w[i] * evaluate(manif, ekr, pt)
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
    ipc[finalpartial] .= getObservability(mkd, true)

    return manikde!(newM, nPts, u0; bw, partial = finalpartial, observability = ipc)
end



## ========================================================================
## Marginalization and partials
## ======================================================================== 


# override
marginal(
    hode::HomotopyDensity,
    partl::AbstractVector{<:Integer},
) = HomotopyDensity(hode, partl)





function _getFieldPartials(
    mkd::HomotopyDensity,
    field::Function,
    aspartial::Bool = true,
)
    partial = getPartial(mkd)
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


"""
    $SIGNATURES

Return true if this HomotopyDensity is a partial.
"""
isPartial(hode::HomotopyDensity) = !isnothing(getPartial(hode))






#
