# define the api for users

"""
    $SIGNATURES

Approximate the pointwise the product of functionals on manifolds using KernelDensityEstimate.

Notes:
- Always pass full beliefs, for partials use for e.g. `partialDimsWorkaround=[1;3;6]`
- Can also multiply different partials together

Example
-------
```julia
# setup
M = TranslationGroup(3)
N = 75
p = manikde!(M, [randn(3) for _ in 1:N])
q = manikde!(M, [randn(3) .+ 1 for _ in 1:N])

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q])

# direct histogram plot
using Gadfly
plot( x=getPoints(pq)[1,:], y=getPoints(pq)[2,:], Geom.histogram2d )

# TODO, convenient plotting (work in progress...)
```
"""
function manifoldProduct(
    ff::AbstractVector{<:HomotopyDensity};
    makeCopy::Bool = false,
    ndims::Integer = maximum([0; Ndim.(ff)]),
    N::Integer = maximum([0; Npts.(ff)]),
    selectedLabels::Vector{Vector{Int}} = Vector{Vector{Int}}(),
    _labelsChoosen_pp::Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}} = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N),
    MC::Int = 3
)
    #
    mani = getManifold(ff[1])

    # check quick exit
    if 1 == length(ff)
        # @show Ndim(ff[1]), Npts(ff[1]), getPoints(ff[1],false)[1]
        return (makeCopy ? x -> deepcopy(x) : x -> x)(ff[1])
    end

    partialDimMask = Vector{BitVector}(undef, length(ff))
    for (k, md) in enumerate(ff)
        partialDimMask[k] = ones(Int, ndims) .== 1
        if isPartial(md)
            for i = 1:ndims
                if !(i in getPartial(md))
                    partialDimMask[k][i] = false
                end
            end
        end
    end

    beliefs = (s -> s.belief).(ff)
    lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(
        mani, 
        beliefs,
        MC;
        _labelsChoosen_pp,
    )

    # push final label selections onto selected`Labels
    resize!(selectedLabels, N)
    for i = 1:N
        selectedLabels[i] = Int[]
        for j = 1:length(ff)
            # k = length(getPoints(ff[j]))
            # @info "HERE" i j lbls
            push!(selectedLabels[i], lbls[i][j])
        end
    end

    # FIXME, this collapses duplicate labels without resampling -- i.e. problem length(posterior) <= N
    lbls_ = unique(lbls)
    N_ = length(lbls_)
    weights = 1 / N .* ones(N_)
    # increase weight of duplicates
    if N_ < N
        for (i, lb_) in enumerate(lbls_)
            idxs = findall(==(lb_), lbls)
            weights[i] = weights[i] * length(idxs)
        end
    end

    post = ApproxManifoldProducts.calcProductKernelsBTLabels(
        mani,
        beliefs,
        lbls_,
        false;
        weights,
    ) # ?? was permute=false?

    # NOTE, resulting tree might not have N number of data points 
    mtr12 = ApproxManifoldProducts.buildTree_Manellic!(mani, post)
    return ManifoldKernelDensity(
        mtr12,
        nothing,
    )

end

# FIXME, this product does not handle combinations of different partial beliefs properly yet
function *(PP::AbstractVector{H}) where {H <: HomotopyDensity}
    return manifoldProduct(PP)
end

function *(P1::HomotopyDensity, P2::HomotopyDensity, P_...) 
    return manifoldProduct([P1; P2; P_...])
end


#
