


"""
    $SIGNATURES

Calculate one product of proposal kernels, as defined  BTLabels.
"""
function calcProductKernelBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector{<:HomotopyDensity},
    labels_sampled::AbstractVector{<:Integer},
    looidx::Union{Int, Nothing} = nothing,
    propIdxs_Gibbs::AbstractVector{<:Integer} = 1:length(proposals);
    permute::Bool = true, # true because signature is BTLabels
    weight::Real = 1.0,
)
    # select a density label from the other proposals
    prop_and_label = @NamedTuple{propidx::Int, problbl::Int}[]
    for s in setdiff(propIdxs_Gibbs, isnothing(looidx) ? Int[] : Int[looidx;])
        # tuple of which leave-one-out-proposal and its new latest label selection
        push!(prop_and_label, (; propidx=s, problbl=labels_sampled[s]))
    end
    # TODO COVARIANCE CONTINUATION CORRECTION FOR DEPTH OF TREE KERNELS
    components = map(
        pr_lb -> getKernelTree(proposals[pr_lb.propidx], pr_lb.problbl, permute, true),
        prop_and_label,
    )

    # TODO upgrade to tuples
    _μ, _Σ, ipc = calcProductGaussians(M, [components...])
    
    # @show ipc

    # FIXME, inflate any partial results
    _partial = findall(!iszero, ipc)
    __partial = length(_partial) == manifold_dimension(M) ? nothing : _partial
    __partial_ = _tuple(__partial)
    M_, reprl, partl_cb = getManifoldPartial(M, __partial_, _μ)
    return ConcentratedGaussianKernel(_μ, _Σ, weight; partial = __partial_, partl_cb)
end

function calcProductKernelsBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector,
    N_lbl_sets::AbstractVector{<:NTuple},
    permute::Bool = true; # true because signature is BTLabels
    weights = 1 / length(N_lbl_sets) .* ones(length(N_lbl_sets)),
)
    #
    # partials = getKernelTree.(proposals, Ref(1)) .|> _getprl
    # @show _mergepartials(M, partials)
    # T = typeof(getKernelTree(proposals[1], 1)) # FIXME FIXME FIXME for products of partials, not just [1]
    N = length(N_lbl_sets)
    # FIXME abstract vectorT not type-stable
    post = Vector{ConcentratedGaussianKernel}(undef, N) 

    for (i, lbs) in enumerate(N_lbl_sets)
        post[i] = calcProductKernelBTLabels(M, proposals, _makevec(lbs); permute, weight = weights[i])
    end

    return post
end


# TODO why not use a standardized `getChildren`?
function generateLabelPoolRecursive(
    proposals::AbstractVector{<:HomotopyDensity},
    labels_sampled::AbstractVector{<:Integer},
)
    # NOTE at top of tree, selections will be [1,1]
    child_label_pools = Vector{Vector{Int}}()

    # Are all selected labels leaves?
    all_leaves = true
    for _ = 1:length(proposals)
        push!(child_label_pools, Vector{Int}())
    end
    for (o, sel) in enumerate(labels_sampled)
        isleaf = true
        # add interval of left and right children for next scale label sampling
        if isassigned(proposals[o], leftIndex(proposals[o], sel))
            push!(child_label_pools[o], leftIndex(proposals[o], sel))
            isleaf = false
        end
        if isassigned(proposals[o], rightIndex(proposals[o], sel))
            push!(child_label_pools[o], rightIndex(proposals[o], sel))
            isleaf = false
        end
        all_leaves &= isleaf
        if isleaf
            push!(child_label_pools[o], sel)
        end
    end

    return child_label_pools, all_leaves
end

"""
    $SIGNATURES

Notes:
- Advise, 2<=MC to ensure multiscale works during decent transitions (TBD obsolete requirement)
- To force sequential Gibbs on leaves only, use:
  `label_pools = [[(length(getPoints(prop))+1):(2*length(getPoints(prop)));] for prop in proposals]`
- References: 
  - Fourie, D., Leonard, J., 2016, Nonparametric solution to the Bayes tree. IEEE ICRA.
  - Fourie, D., 2017. Multi-modal and Inertial Sensor Solutions for Navigation-type Factor Graphs. MIT/WHOI PhD Thesis.
  - Fourie, D., Leonard, J., 2018. On-Manifold Nonparametric Density Estimation.  IEEE IROS.
  - Sudderth, E.B., Ihler, A.T., Isard, M., Freeman, W.T. and Willsky, A.S., 2010. Nonparametric belief propagation. Communications of the ACM, 53(10), pp.95-103.
"""
function sampleProductSeqGibbsBTLabel(
    M::AbstractManifold,
    proposals::AbstractVector{<:HomotopyDensity},
    MC::Int = 3,
    # pool of sampleable labels
    label_pools::Vector{Vector{Int}} = [[1:1;] for _ in proposals],
    labels_sampled::Vector{Int} = [rand(label_pools[i]) for i in 1:length(proposals)];
    # multiscale_parents = nothing;
    MAX_RECURSE_DEPTH::Int = 24, # 2^24 is so deep
    _labelsChoosen::Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}} = Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}()
)
    # local helpers for partials either vec or nothing
    _leng(s::Nothing) = 0
    _leng(s::Union{<:AbstractVector{<:Integer}, <:Tuple}) = length(s)

    # apply further partials to existing kernel
    # how many incoming proposals
    d = length(proposals)
    propIdxs_Gibbs = 1:d

    _trivial_label_pool = all(length.(label_pools) .== 1)
    # pick the next leave-out proposal
    # TODO, gibbSeq might be different for unbalanced nodes "cross-products" during multiscale
    for _burn = 1:MC, lvout_idx in propIdxs_Gibbs
        # on first pass labels_sampled come from parent-recursive as part of multi-scale (i.e. pre-homotopy) operations
        # calc product of Gaussians from currently selected \LOO-proposals
        lvin_product_tmp = calcProductKernelBTLabels(
            M,
            proposals,
            labels_sampled,
            lvout_idx,
            propIdxs_Gibbs;
            permute = false,
        )
        
        # to find leave-out (LO) resample weights, evaluate leave-in (LI) mean against temporary leavein_product kernel
        lvout_centers = [mean(getKernelTree(proposals[lvout_idx], i, false)) for i in label_pools[lvout_idx]]
        # if lvout_centers are partial, then only evaluate with partial lvin_product_tmp
        lvout_prl = _getprl(getKernelTree(proposals[lvout_idx], label_pools[lvout_idx][1], false))
        lvin_product_tmp_partial = _intersectpartials(M, lvin_product_tmp, lvout_prl)

        # overcome case where no partial overlap exists
        resample_weights = if 0 < _leng(_getprl(lvin_product_tmp_partial))
            resample_weights = evaluateDensityAtPoints(M, lvin_product_tmp_partial, lvout_centers, true)
            # update label-distribution of out-proposal from product of selected LOO-proposal components
            p = Categorical(resample_weights)
            labels_sampled[lvout_idx] = label_pools[lvout_idx][rand(p)]
            resample_weights
        else
            NaN*ones(length(lvout_centers))
        end

        # slightly heavy memory usage to aid DX
        push!(_labelsChoosen, (;
            loo = lvout_idx,
            selected = deepcopy(labels_sampled),
            pool = deepcopy(label_pools),
            catp = deepcopy(resample_weights),
        ))

        # don't have to resample if only one label to choose from
        if _trivial_label_pool && ( lvout_idx == propIdxs_Gibbs[end])
            break
        end
    end

    # construct new label pool for children in multiscale
    child_label_pools, all_leaves = generateLabelPoolRecursive(proposals, labels_sampled)

    # recursively call sampling down the multiscale tree ("pyramid") -- aka homotopy
    # limit recursion to MAX_RECURSE_DEPTH
    # FIXME, final label selection should not be sensitive to being all_leaves.
    if 0 < MAX_RECURSE_DEPTH && !all_leaves
        # @info "Recurse down manellic tree for multiscale product"
        # labels_sampled_copy = deepcopy(labels_sampled)
        labels_sampled = sampleProductSeqGibbsBTLabel(
            M,
            proposals,
            MC,
            child_label_pools;
            # labels_sampled_copy; # randomly sample from new child pool
            MAX_RECURSE_DEPTH = MAX_RECURSE_DEPTH - 1,
            _labelsChoosen,
        )

        # TODO, [circa 2006, Rudoy & Wolfe] detailed balance (Hastings) by rejecting a multiscale decent given simulated or parallel tempering
        # recursive call of sampleProductSeqGibbsBTLabel but with same parameters as this function invokation, aka reject the decend
    end

    #
    return labels_sampled
end


function sampleProductSeqGibbsBTLabels(
    M::AbstractManifold,
    proposals::AbstractVector{<:HomotopyDensity},
    MC::Int = 3,
    N::Int = round(Int, mean(Npts.(proposals))), # FIXME use getLength or length of proposal (not getPoints)
    label_pools = [[1:1;] for _ in proposals];
    _labelsChoosen_pp::Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}} = Vector{Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}}(undef, N)
)
    #
    d = length(proposals)
    posterior_labels = Vector{NTuple{d, Int}}(undef, N)

    for i = 1:N
        _labelsChoosen_pp[i] = Vector{@NamedTuple{loo::Int64, selected::Vector{Int64}, pool::Vector{Vector{Int64}}, catp::Vector{Float64}}}()
        posterior_labels[i] =
            tuple(sampleProductSeqGibbsBTLabel(M, proposals, MC, label_pools; _labelsChoosen = _labelsChoosen_pp[i])...)
    end

    return posterior_labels
end
