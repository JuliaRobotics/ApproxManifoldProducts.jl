

## ==========================================================================================
## UX/DX Convenience functions
## ==========================================================================================


# FIXME, heavy legacy -- update this to a prettier show of modern HomotopyDensity
function Base.show(io::IO, hode::HomotopyDensity{H, P, HL, HT}) where {H, P, HL, HT}
    N = Npts(hode)
    printstyled(io, "HomotopyDensity{"; bold = true, color = :blue)
    println(io)
    printstyled(io, "    partial"; bold = true, color = :magenta)
    print(io, " = ", getPartial(hode), ",")
    println(io)
    printstyled(io, "    M"; bold = true, color = :magenta)
    print(io, " = ", typeof(getManifold(hode.representationkind)), ",")
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
    @assert Npts(hode) == length(hode.elements) "show(::HomotopyDensity,) noticed a data size issue, expecting N$(Npts(hode)) == length(.elements)$(length(hode.elements))"
    if 0 < Npts(hode)
        println(io, "  .elements[1:]   :  ", hode.elements[1], " ... ", hode.elements[end])
        println(io, "  .weights[1:]:  ", hode.weights[1], " ... ", hode.weights[end])
        printstyled(io, "     (uniwt)  :   ", uniWT(hode); color = :light_black)
        println(io)
        print(io, "  .structure[1][-]:  ")
        printstyled(io, hode.structure[1][1], " ... ", hode.structure[1][end]; color = :light_black)
        println(io)
        print(io, "  .tkernels[") # " __see below__"; color=:light_black)
        if 0 < Npts(hode)
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
## HomotopyDensity constructorhelper functions
## ==========================================================================================


function HomotopyDensity(
    bel::HomotopyDensity,
    partial_::L;
    observability::AbstractVector{<:Real} = bel.observability,
) where {L <: Union{<:AbstractVector{<:Integer}, <:Tuple}}
    #
    N = Npts(bel)
    partial = _tuple(partial_)
    mani = getManifold(bel)
    partl = _intersect(getPartial(bel), partial)
    M_, reprl, partl_cb = getManifoldPartial(
        mani, 
        partl, 
        bel.elements[1],
    )
    if length(partl) != manifold_dimension(mani)
        # update representation kind to have correct partials
        _partialrepr(::HomotopyRepresentation{M, L, K, D}) where {M, L, K, D} = HomotopyRepresentation{M, partl, K, D}(mani)
        representationkind = _partialrepr(bel.representationkind)
        # assuming there are tree and leaf nodes at [1]...
        _tkT() = _intersectpartials(mani, getKernelTree(bel, 1), partial) |> typeof
        _lkT() = _intersectpartials(mani, getKernelLeaf(bel, 1), partial) |> typeof
        tree_kernels  = Vector{_tkT()}(undef, length(bel.tree_kernels))
        leaf_kernels  = Vector{_lkT()}(undef, length(bel.leaf_kernels))
        tkm = (s->isassigned(bel.tree_kernels, s)).(1:length(bel.tree_kernels))
        lkm = (s->isassigned(bel.leaf_kernels, s)).(1:length(bel.leaf_kernels))
        tree_kernels_ = view(tree_kernels, tkm)
        leaf_kernels_ = view(leaf_kernels, lkm)
        tree_kernels_ .= (s->_intersectpartials(mani, s, partial, partl_cb)).(view(bel.tree_kernels, tkm))
        leaf_kernels_ .= (s->_intersectpartials(mani, s, partial, partl_cb)).(view(bel.leaf_kernels, lkm))
        # update minors detail to have correct partials
        nzs, _ = SparseArrays.findnz(bel.minors_detail)
        for i in nzs
            cv = bel.minors_detail[i].mat
            dummy = bel.elements[1]
            cg = ConcentratedGaussianKernel(dummy, cv)
            cg_ = _intersectpartials(mani, cg, partial, partl_cb)
            cv_ = cov(cg_)
            bel.minors_detail[i] = PDMat(SMatrix{size(cv_)..., Float64}(cv_))
        end
        # update density to have correct partials
        # partial = _getprl(eltype(tree_kernels))
        bel_ = HomotopyDensity(;
            representationkind,
            observability,
            elements = bel.elements,
            weights = bel.weights,
            structure = bel.structure,
            leaf_kernels,
            tree_kernels,
            minors_detail = bel.minors_detail,
        )

        # call the constructor direct
        return bel_
    else
        # full manifold, i.e. partial=nothing
        return bel
    end
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
    manif::M,
    r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
    N = length(r_PP),
    weights::AbstractVector{<:Real} = ones(N) .* (1 / N),
    kernel = ConcentratedGaussianKernel,
    kernel_bw = nothing, # TODO
    partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {M <: AbstractManifold, P <: AbstractArray}
    #
    
    D = manifold_dimension(manif)
    CV = SMatrix{D, D, Float64, D * D}(diagm(ones(D)))
    prlcb = if isnothing(partl_cb) && !isnothing(partial)
        M_, reprl, cb = getManifoldPartial(manif, partial)
        cb
    else
        partl_cb
    end
    
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)
    _legacybw(::Nothing) = CV
        
    lCV = _legacybw(kernel_bw)
    tknlT = kernel(r_PP[1], CV; partial=_tuple(partial), partl_cb=prlcb) |> typeof
    lknlT = kernel(r_PP[1], lCV; partial = _tuple(partial), partl_cb=prlcb) |> typeof

    # leaf kernels
    lkern = Vector{lknlT}(undef, N)
    for i = 1:N
        nkr = kernel(r_PP[i], lCV; partial = _tuple(partial), partl_cb=prlcb)
        lkern[i] = nkr
    end
    tkern = Vector{tknlT}(undef, N)

    representationkind = HomotopyRepresentation{
        M,
        _tuple(partial),
        ConcentratedGaussianKernel,
        MajorMaxDepth{3},
    }(manif)

    # TODO consolidate w legacy kernel_bw
    d = manifold_dimension(getManifold(representationkind))
    minors_detail = SparseArrays.sparsevec(Dict(
        1 => PDMat(SMatrix{d,d,Float64}(cov(lkern[1]))),
    ), 1) # assume size 1 during refactor -- i.e. universal bandwidth at leaves

    _hode = HomotopyDensity{
        typeof(representationkind),
        eltype(r_PP),
        lknlT,
        tknlT,
        eltype(minors_detail),
    }(;
        representationkind,
        elements = r_PP,
        weights,
        # TODO deprecating fields below
        leaf_kernels = lkern,
        tree_kernels = tkern,
        minors_detail,
    )

    #
    tosort_leaves = buildTree_Manellic!(
        _hode,
        1, # start at root
        1, # spanning all data
        N; # to end of data
        kernel,
        kernel_bw,
        partial = _tuple(partial),
        partl_cb = prlcb,
    )

    return tosort_leaves
end


function HomotopyDensity_legacy(;
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
  manifold::M, 
  elements::Vector{P},
  leaf_kernels::Vector{HL},
  tree_kernels::Vector{HT},
  kernel_bw = nothing,
  kw...
) where {M, P, HL, HT}
    _legacybw(s::AbstractMatrix) = s
    _legacybw(s::AbstractVector) = diagm(s)
    _legacybw(::Nothing) = LinearAlgebra.I
        
    lCV = _legacybw(kernel_bw)

    representationkind = HomotopyRepresentation{
        M, 
        partial, 
        ConcentratedGaussianKernel, 
        MajorMaxDepth{3}
    }(manifold)

    d = manifold_dimension(getManifold(representationkind))
    minors_detail = SparseArrays.sparsevec(Dict(
        1 => PDMat(SMatrix{d,d,Float64}(lCV)),
    ), 1)

    HomotopyDensity{
        typeof(representationkind),
        P, 
        HL,
        HT,
        eltype(minors_detail),
    }(;
        representationkind,
        elements,
        leaf_kernels,
        tree_kernels,
        minors_detail,
        kw...
    )
end

function HomotopyDensity(
  hode::HomotopyDensity;
  partial::Union{Nothing, <:Tuple, AbstractVector{<:Integer}} = nothing,
)
  partl = getPartial(hode)
  _partl = _intersect(partial, partl)
  HomotopyDensity_legacy(;
    partial = _partl,
    manifold = getManifold(hode),
    elements = hode.elements,
    leaf_kernels = hode.leaf_kernels,
    tree_kernels = hode.tree_kernels,
    weights = getWeights(hode),
    structure = hode.structure,
    observability = hode.observability,
  )
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



## ==========================================================================================
## a few utilities
## ==========================================================================================

"""
    $SIGNATURES

Alias for overloaded `Statistics.mean`.
"""
calcMean(mkd::HomotopyDensity, aspartial::Bool = true) = mean(mkd, aspartial)


function Statistics.mean(mkd::HomotopyDensity, aspartial::Bool = true; kwargs...)
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
Npts(hode::HomotopyDensity) = length(hode.elements)
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
    pts = permute ? view(hode.elements, hode.structure[1]) : hode.elements

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
    hode::HomotopyDensity,
    aspartial::Bool = true,
)
    partl = getPartial(hode)
    bws = (s->getBW(getKernelLeaf(hode, s))).(1:Npts(hode))
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



getPointRepr(x::HomotopyDensity) = eltype(x.elements) # TODO use HomotopyDensity{T} style instead
function getManifold(x::HomotopyDensity, aspartial::Bool = false)
    return if !aspartial
        getManifold(x.representationkind)
    else
        M_, _, _ = getManifoldPartial(getManifold(x), getPartial(x), x.elements[1])
        M_
    end
end



function getObservability(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> x.observability, aspartial)
end

function getBandwidth(mkd::HomotopyDensity, aspartial::Bool = true)
    return _getFieldPartials(mkd, x -> getBW(x)[1], aspartial)
end


# TODO check that partials / marginals are sampled correctly
function sample(belief::HomotopyDensity, N::Integer = 1)
    # get legacy matrix of coordinates and selected labels
    coords, lbls = sample(belief, N)
    # pack samples into vector of point type P
    vecP = Vector{eltype(belief.elements)}(undef, N)
    for j = 1:N
        vecP[j] = makePointFromCoords(getManifold(belief), view(coords, :, j), belief.elements[1])
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
        observability = x.observability,
    )
end


function updateBandwidths(
    hode::HomotopyDensity{H, P, HL}, 
    bws;
    partl_cb::Union{Nothing, <:Function} = nothing,
) where {H, P, HL}
    #
    _getBW(s::Float64, ::Int) = [s;;]
    _getBW(s::AbstractVector{<:Real}, ::Int) = s
    _getBW(s::AbstractMatrix{<:Real}, ::Int) = s
    _getBW(s::AbstractVector{<:AbstractArray}, _i::Int) = s[_i]

    N = Npts(hode)

    (nzi,_) = SparseArrays.findnz(hode.minors_detail)

    leaf_kernels = Vector{HL}(undef, N)
    for (i, lk) in enumerate(hode.leaf_kernels)
        nkl = ConcentratedGaussianKernel(lk; Σ = _getBW(bws, i), partl_cb)
        leaf_kernels[i] = nkl # updateKernelBW(lk, _getBW(bws, i))
        # new replacement field instead of .leaf_kernels
        if i in nzi
            cv = cov(nkl)
            hode.minors_detail[i] = PDMat(SMatrix{size(cv)...,Float64}(cv))
        end
    end
    kind = getManifold(hode) 
    representationkind = HomotopyRepresentation{
        typeof(kind),
        getPartial(hode),
        ConcentratedGaussianKernel,
        MajorMaxDepth{3},
    }(kind)

    return HomotopyDensity(;
        representationkind,
        elements = hode.elements,
        weights = hode.weights,
        structure = hode.structure,
        leaf_kernels,
        tree_kernels = hode.tree_kernels,
        minors_detail = hode.minors_detail,
    )
end

"""
    $SIGNATURES
    
For Manellic tree parent kernels, what is the 'smallest' and 'biggest' covariance.

Notes:
- Thought about `det` for covariance volume but long access of pancake (smaller volume) is not minimum compared to circular covariance. 
"""
function getBandwidthSearchBounds(hode::HomotopyDensity)
    upper = cov(hode.tree_kernels[1])

    #FIXME isdefined does not work as expected for hode.tree_kernels, so using length-1 for now
    # this will break if number of points is not a power of 2. 
    
    lower_diag = diag(cov(hode.tree_kernels[1]))
    for i in 2:(length(hode.tree_kernels) - 1)
        # FIXME use consolidated getKernelTree instead
        if isassigned(hode.tree_kernels, i)
            hdg = hcat(lower_diag, diag(cov(hode.tree_kernels[i])))
            lower_diag = minimum(hdg; dims = 2)
        end
        # lower_diag = minimum(hcat(lower_diag, diag(cov(hode.tree_kernels[i]))); dims = 2)
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


getPartial(hode::HomotopyDensity) = getPartial(hode.representationkind)



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
