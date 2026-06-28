
abstract type AbstractBinaryTreeDensity <: AbstractHomotopyTopology end
const BinaryTreeDensity = AbstractBinaryTreeDensity


struct BinaryTruncFixedDepth{N} <: AbstractBinaryTreeDensity end
# struct BinaryInjectivityThres{N} <: AbstractBinaryTreeDensity{N} end
# struct PrincipalEnergyThres{N} <: AbstractHomotopyTopology{N} end

struct PartialNoSerde <: AbstractPartialTrait end


struct HomotopyReprLive{
  T <: Union{<:StateType, <:AbstractManifold},
  # Nothing and Tuple are legacy before DFG v1.0, trying to avoid serde of old partials devoid of serde design
  L <: Union{<:AbstractPartialTrait, Nothing, <:Tuple}, 
}
  topologykind::AbstractHomotopyTopology  # bitmap, jpeg, png
  formkind::AbstractDensityForm           # RGB24, YCbCr, fullcov, uppercov, LieExpGaussianWrappedKind, ConcentrGaussKernelKind
  statekind::T                            # Position{2}
  partial::L
end
const HomotopyRepr = Union{<:HomotopyReprLive, <:HomotopyReprDFG}


@kwdef struct HomotopyDensityLive{
  H <: HomotopyRepr, # serde friendly when using DFG.statekind representation, but also supports Manifolds.jl direclty
  # parameters below auto generate during JSON lift, only above needs to be serde friendly
  P, # serde relies on DFG statekind mechanism, does not guarantee serde when directly using Manifolds wo DFG.statekind
  ME,  # Major elements can be points or eigen vectors etc.
  MJ,
  MI,
}
  reprkind::H
  observability::Vector{Float64} = zeros(manifold_dimension(getManifold(reprkind)))
  principal_coeffs::Vector{Float64} = Vector{Float64}(undef, getMajorsLength(reprkind))
  # FIXME, future proof such that ME != P, possibly using affine_matrix
  principal_elements::Vector{ME} = Vector{P}(undef, getMajorsLength(reprkind))
  principal_forms::Vector{MJ} = Vector{Matrix{Float64}}(undef, getMajorsLength(reprkind))
  """
  Store minor details such as leaf bandwidth or eigenvectors associated with minor eigenvalues.
  - When lifted for compute efficiency, this field is likely to hold something like PDMats.
  - When lowered or for serde, this field is likely to hold Dict{Int, Vector{Float64}}.
  """  
  points::Vector{P}
  weights::Vector{Float64} = Vector{Float64}(ones(length(points))) ./ length(points)
  trailing_forms::SparseArrays.SparseVector{MI, Int} = SparseArrays.sparsevec(
    Dict(1 => SMatrix{Float64}(I, manifold_dimension(getManifold(reprkind)), manifold_dimension(getManifold(reprkind))),),
    1
  )
  """ 
  Geometric points permute field, allows fast binary tree operations and geometric points splits for manellic (ball) trees. 
  - Geometric split reqs at least 2*(N+1)-1 points -- e.g. when nodes have only right children, points=[1,2,-3].
  """
  structure::SparseArrays.SparseVector{Vector{Int}, Int} = SparseArrays.sparsevec(
    Dict(1 => collect(1:length(points))), 
    5*(length(points)) # large buffer space where impact on resources mitigated via sparsevec
  )
end
# Solve DataLevel 3.5
const HomotopyDensity = Union{<:DistributedFactorGraphs.HomotopyDensityDFG, <:HomotopyDensityLive}



function HomotopyDensityLive(hode::HomotopyDensityDFG)
  _second(::Pair{K, V}) where {K, V} = V
  # convert trailing details from Dict{Int, Matrix{Float64}} to SparseVector{Int, SMatrix{N,N}} for live representation
  tdks = keys(hode.trailing_forms)
  tdvs = values(hode.trailing_forms)
  rc = 0 < length(tdvs) ? size(tdvs[1]) : (0, 0)
  trailing_forms = SparseArrays.sparsevec(Dict{Int,_second(eltype(tdvs))}(
    tdks .=> SMatrix{rc...}.(tdvs)
  ))
  
  return HomotopyDensityLive(
    reprkind = hode.reprkind,
    observability = hode.observability,
    points = hode.points,
    weights = hode.weights,
    principal_coeffs = hode.principal_coeffs,   # FIXME, convert type
    principal_elements = hode.principal_elements, # FIXME, convert type 
    principal_forms = hode.principal_forms,  # FIXME, convert type
    trailing_forms,
    structure = SparseArrays.sparsevec(hode.structure.nzidx, hode.structure.nzval),
  )
end
HomotopyDensityDFG(hode::HomotopyDensityLive) = HomotopyDensityDFG(
  reprkind = HomotopyReprDFG(hode.reprkind),
  observability = hode.observability,
  points = hode.points,
  weights = hode.weights,
  principal_coeffs = hode.principal_coeffs,
  principal_elements = hode.principal_elements,
  principal_forms = begin
    pf = Vector{Matrix{Float64}}(undef, length(hode.principal_forms))
    for i in 1:length(hode.principal_forms) 
      if isassigned(hode.principal_forms, i)
        pf[i] = Matrix(hode.principal_forms[i])
      end
    end 
    pf
  end,
  trailing_forms = SparseArrays.sparsevec(hode.trailing_forms.nzind, Matrix.(hode.trailing_forms.nzval)), # likely diagonal covs
  structure = hode.structure,
)



## ==========================================================================================
## HomotopyDensity constructorhelper functions
## ==========================================================================================


# TODO consolidate with common HomotopyDensity/replace constructor helper
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

function HomotopyRepr(
  repr::HomotopyRepr = HomotopyReprLive(
    BinaryTruncFixedDepth{3}(),
    ConcentratedGaussianKernel(),
    TranslationGroup(1),
    nothing,    
  );
  topologykind::AbstractHomotopyTopology = getTopologyKind(repr),
  formkind::AbstractDensityForm = getFormKind(repr),
  statekind::Union{<:AbstractStateType, <:AbstractManifold} = getStateKind(repr),
  partial = getPartial(repr), # FIXME
) 
  return HomotopyReprLive(
    topologykind,
    formkind,
    statekind,
    partial,
  )
end


function HomotopyReprDFG(
  hr::HomotopyReprLive;
  topologykind = hr.topologykind,
  formkind = hr.formkind,
  statekind = hr.statekind,
  partial = hr.partial,
)
  if !(statekind isa AbstractStateType)
    throw(ArgumentError("JSON.jl serde using DFG representation requires statekind to be a StateType -- you can easily expand serde support for your manifold with DistributedFactorGraph.@defStateType.  Alternatively, Homotopy*Live supports direct use of Manifolds.jl types without serde: $statekind"))
  end
  if !(partial isa Nothing)
    # throw(ArgumentError("JSON.jl serde using DFG representation does not currently support partials -- you can easily expand serde support for your partial with DistributedFactorGraph.@defPartialTrait.  Alternatively, Homotopy*Live supports direct use of partials without serde."))
    @error("JSON.jl serde using DFG representation does not currently support partials -- you can easily expand serde support for your partial with DistributedFactorGraph.@defPartialTrait.  Alternatively, Homotopy*Live supports direct use of partials without serde.")
  end
  return DistributedFactorGraphs.HomotopyReprDFG(
    topologykind,
    formkind,
    statekind,
    PartialNoSerde(), # TODO
  )
end

function HomotopyReprLive(
  hr::HomotopyReprDFG;
  topologykind = hr.topologykind,
  formkind = hr.formkind,
  statekind = hr.statekind,
  partial = nothing, # TODO
)
  return HomotopyReprLive(
    topologykind,
    formkind,
    statekind,
    partial,
  )
end

## =========================================================================================
## Explicit converts
## =========================================================================================


convert(::Type{<:HomotopyReprLive}, src::HomotopyReprDFG) = HomotopyReprLive(src)
convert(::Type{<:HomotopyReprDFG}, src::HomotopyReprLive) = HomotopyReprDFG(src)


# overload Base.convert for easy conversion between live and hold representations
convert(::Type{<:HomotopyDensityLive}, src::HomotopyDensityDFG) = HomotopyDensityLive(src)
convert(::Type{<:HomotopyDensityDFG}, src::HomotopyDensityLive) = HomotopyDensityDFG(src)


# consolidate in IIF had weird cases requiring conversion between MArray and SArray types.  This convert became the easiest
# trivial case with no convert between the same P point types
convert(::Type{<:HomotopyDensityDFG{S,P}}, src::HomotopyDensityDFG{S,P}) where {S,P} = src
function convert(
  ::Type{<:HomotopyDensityDFG{S,sP}}, 
  src::HomotopyDensityDFG{S,P}
) where {S, sP, P}
  # @info "convert" string(src.principal_elements) string(src.points) string(src.trailing_forms)

  principal_elements = Vector{sP}(undef, length(src.principal_elements))
  for i in 1:length(src.principal_elements)
    if isassigned(src.principal_elements, i)
      principal_elements[i] = convert(sP, src.principal_elements[i])
    end
  end
  points = Vector{sP}(undef, length(src.points))
  for i in 1:length(src.points)
    if isassigned(src.points, i)
      points[i] = convert(sP, src.points[i])
    end
  end

  return HomotopyDensityDFG{S, sP}(;
    reprkind = src.reprkind,
    observability = src.observability,
    principal_coeffs = src.principal_coeffs,
    principal_elements,
    principal_forms = src.principal_forms,
    points,
    weights = src.weights,
    trailing_forms = src.trailing_forms,
    structure = src.structure,
  )
end

