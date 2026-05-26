

# forcing ProductManifold to use ArrayPartition as accompanying representation
const _PartiableRepresentationProduct = Union{
    Nothing, 
    <:ArrayPartition
}
# forcing ProductManifold to use ArrayPartition as accompanying representation
const _PartiableRepresentationFlat{T} = Union{
    Nothing, 
    <:AbstractVector{T}
}
# More general representation for Manifold Factors or Groups
const _PartiableRepresentation = Union{
    <:_PartiableRepresentationProduct,
    <:_PartiableRepresentationFlat,
    <:AbstractMatrix,
}


## weird internal functions for handling partials as vectors or tuples of coordinate indices.

# FIXME, a better solution is needed for sqrt_iΣ, especially for partials.
_sqrt_Σ(k::ConcentratedGaussianKernel{L}) where {L} = _getpartial(L, sqrt_Σ(k))
_sqrt_Σ(k::ConcentratedGaussianKernel{Nothing}) = sqrt_Σ(k)
_sqrt_iΣ(k::ConcentratedGaussianKernel{L}) where {L} = inv(sqrt(_getpartial(L, cov(k))))
_sqrt_iΣ(k::ConcentratedGaussianKernel{Nothing}) = sqrt_iΣ(k)

_forcestatic(s::SVector) = s
_forcestatic(s::AbstractVector) = SVector(s...)
_forcestatic(s::AbstractMatrix) = SMatrix{size(s)...}(s)
_forcestatic(s::ArrayPartition) = ArrayPartition(_forcestatic.(s.x)...)

# partials sometimes require values to be masked out as Inf or NaN, TBD if pure stack allocations can be used
_forcemutable(s::MMatrix) = s
_forcemutable(s::AbstractMatrix) = MMatrix{size(s)...}(s)
_forcemutable(s::MVector) = s
_forcemutable(s::AbstractVector) = MVector{length(s)}(s)
_forcemutable(s::ArrayPartition) = ArrayPartition(_forcemutable.(s.x)...)


# kernels explicitly change to partial definition via tuples (for clarity during development) 
_tuple(p::Nothing) = p
_tuple(p::Tuple) = p
_tuple(p::AbstractVector{<:Integer}) = tuple(p...)

_makevec(s::Nothing) = s
_makevec(w::AbstractVector) = w
_makevec(w::Tuple) = [w...]

_getprl(::Type{<:ConcentratedGaussianKernel{partial}}) where partial = partial
_getprl(::ConcentratedGaussianKernel{partial}) where partial = partial

_getpartial(  ::Nothing, s) = s
_getpartial(_pr::Tuple, v::AbstractVector) = view(v, SVector(_pr...))
_getpartial(_pr::Tuple, v::AbstractMatrix) = view(v, SVector(_pr...), SVector(_pr...))
_getpartial(_pr::Tuple, m::AbstractManifold) = getManifoldPartial(m, _makevec(_pr))[1]
_getpartial(partial::AbstractVector{Int}, s) = _getpartial(_tuple(partial), s)

_viewprl(s::AbstractArray, partial::Nothing) = s
_viewprl(s::AbstractArray, partial::Tuple) = _viewprl(s, _makevec(partial))
_viewprl(s::AbstractVector, partial::AbstractVector) = view(s, partial)
_viewprl(s::AbstractMatrix, partial::AbstractVector) = view(s, partial, partial)

# apply further partials to existing kernel, i.e. intersect with existing partials if they exist, otherwise just apply new partial
_intersect(a::Nothing,::Nothing) = a
_intersect(::Nothing, b) = b
_intersect(a, ::Nothing) = a
_intersect(a, b) = tuple(intersect(a,b)...)
function _intersectpartials(
    M::AbstractManifold, 
    k::ConcentratedGaussianKernel, 
    prl::Union{Nothing, <:Tuple, <:AbstractVector{Int}},
    _partl_cb::Union{Nothing, <:Function} = nothing,
)
    prlA = _getprl(k)
    prlB = _tuple(prl)
    partial = _intersect(prlA, prlB)
    μ = mean(k) # this is on-manifold
    Σ2 = cov(k) # this is on tangent
    partl_cb = if !isnothing(partial) && isnothing(_partl_cb)
        plM, plrep, partl_cb_ = getManifoldPartial(M, partial, μ; doError = false)
        partl_cb_
    else
        _partl_cb
    end
    return ConcentratedGaussianKernel(
        μ, 
        Σ2; 
        partial,
        partl_cb,
    )
end

function _mergepartials(
    M::AbstractManifold,
    partials::AbstractVector
)
    d = manifold_dimension(M)
    prlm = zeros(Int,d)
    for pl in partials
        if isnothing(pl)
            prlm .+= 1
        else
            for i in pl
                prlm[i] += 1
            end
        end
    end
    partial = findall(!iszero, prlm)
    if length(partial) != d
        return tuple(partial...)
    end
    return nothing
end


# TODO this should be a public method relating to getManifold
function _getManifoldFullOrPart(hode::HomotopyDensity, aspartial::Bool = true)
    if aspartial && isPartial(hode)
        getManifoldPartial(getManifold(hode), getPartial(hode))
    else
        getManifold(hode)
    end
end


## ---------------------

function _invs(
    Σ_::Union{<:AbstractVector{S}, <:NTuple{N, S}}; 
    partials::Union{<:AbstractVector, <:Tuple}
) where {N, S <: AbstractMatrix{<:Real}}
    d = size(Σ_[1])[1]
    infs = diagm(MVector{d}([Inf for _ in 1:d]))
    Λs = [deepcopy(infs) for _ in 1:length(Σ_)]
    for (i,s) in enumerate(Σ_)
        dst = _viewprl(Λs[i], partials[i]) 
        dst .= inv(_viewprl(s, partials[i]))
    end
    return Λs
end

function _mean(
    M::AbstractManifold, 
    v::Union{<:AbstractVector{P}, <:NTuple{N, P}}; 
    partials::Union{<:AbstractVector, <:Tuple}
) where {N, P <: AbstractArray}
    # hack during dev testing
    s = if all(isnothing.(partials))
        mean(M, _makevec(v))
    elseif P <: AbstractVector
        d = manifold_dimension(M)
        mn = MVector{d}([0.0 for _ in 1:d])
        cu = MVector{d}([0 for _ in 1:d])
        for (s,pl) in zip(v,partials)
            _mn = _viewprl(mn, pl)
            _mn .+= _viewprl(s, pl)
            _cu = _viewprl(cu, pl)
            _cu .+= 1
        end
        (mn ./ cu)
    else
        error("TODO calc partial mean of non-vector manifold types $(M), v isa $(typeof(v)), given $(partials)")
    end

    return s
end


## ================================================================================
## COMMON UTILS FOR PARTIAL MANIFOLDS

function _checkManifoldPartialDims(
    M::AbstractManifold,
    partial_::Union{<:AbstractVector{Int}, <:Tuple},
    offset::Base.RefValue{Int},
    doError::Bool = true,
)
    partial = _tuple(partial_)
    d = manifold_dimension(M)
    full = 1:d
    mask = 0 .== 1:d
    offp = partial .- offset[] 
    for i in 1:d
        if full[i] in offp
            mask[i] = true
        end
    end
    # mask = 0 .< (partial .- offset[]) .<= manifold_dimension(M)
    doError &&
        !any(mask) &&
        error(
            "Unknown partial=$(partial .- offset[]) over dimensions=$(manifold_dimension(M)) of manifold $M",
        )
    return mask
end

## EXTRACT REPRESENTATION PARTIALS

# do nothing case
_getReprPartial(M::MB.AbstractManifold, ::Nothing, w...; kw...) = nothing

# drop back to regular vector
function _getReprPartial(
    M::MB.AbstractManifold,
    repr::AbstractVector{T},
    partial_::Union{<:AbstractVector{Int}, <:Tuple}, # total partial from user over all Factors
    offset::Base.RefValue{Int} = Ref(0),
    mask::BitVector = _checkManifoldPartialDims(M, partial_, offset, doError),
) where {T <: Number}
    partial = _tuple(partial_)
    ret = zeros(T, sum(mask))
    for (i, p) in enumerate(partial .- offset)
        ret[i] = repr[p]
    end
    return ret
end

function _getReprPartial(
    M::Union{
        <:typeof(SpecialOrthogonalGroup(2)),
        <:Manifolds.Rotations{TypeParameter{Tuple{2}}},
    },
    repr::AbstractMatrix{T},
    partial::AbstractVector{Int}, # total partial from user over all Factors
    offset::Base.RefValue{Int} = Ref(0),
    mask::BitVector = _checkManifoldPartialDims(M, partial, offset, false),
) where {T <: Number}
    @assert sum(mask) == 1 "Can only return the same point represenation matrix for SpecialOrthogonalGroup(2) / Rotations(2)"
    return repr
end


function _rotateCoordsPartial(
    M::AbstractLieGroup,
    r_CCp::AbstractVector,
    ax_R_r::AbstractMatrix;
    partial::Union{Nothing, <:Tuple} = nothing,
)
    _unrollpartial(::Nothing) = LinearAlgebra.I
    _unrollpartial(p::Tuple) = begin
        m = zeros(Int,manifold_dimension(M))
        m[[p...]] .= 1
        return m
    end
    _unrollpartial(p::ArrayPartition) = error("TODO _unrollpartial for ArrayPartition")
    _ = _unrollpartial(partial) # FIXME
    _ax_R_r = _forcemutable(ax_R_r)
    # remove Nans
    for i in axes(_ax_R_r, 1)
        for j in axes(_ax_R_r, 2)
            if !isnothing(partial) && (!(i in partial) || !(j in partial))
                # default values for inactive elements of rotation matrix
                _ax_R_r[i,j] = i == j ? 1.0 : 0.0
            end
            # else leave row and column unchanged
        end
    end

    # rotate coordinates
    return map(r_CCp) do r_Cp
        _r_Cp = _forcemutable(r_Cp)
        for j in 1:length(_r_Cp)
            if !isnothing(partial) && !(j in partial)
                # default values for inactive coordinates
                _r_Cp[j] = 0.0
            end
            # else leave coordinate unchanged
        end
        _ax_R_r * _r_Cp
    end
end


## ================================================================================
## EXTRACT PARTIAL MANIFOLD

# This is the trivial do-nothing case
function getManifoldPartial(
    M::AbstractManifold,
    partial::Nothing,
    repr::_PartiableRepresentation = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    kw...,
)
    offset[] += manifold_dimension(M)
    return (M, repr, (s)->s)
end


function getManifoldPartial(
    M::Union{<:Manifolds.Euclidean{Tuple{N}}, <:TranslationGroup},
    partial_::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentationFlat{T} = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
) where {N, T <: Number}
    partial = _tuple(partial_)
    mask = _checkManifoldPartialDims(M, partial, offset, doError)
    offset[] += manifold_dimension(M)
    len = sum(mask)
    repr_p = repr === nothing ? nothing : zeros(T, len)
    # EXPERIMENTAL, use lambda to construct partial lookup
    return (TranslationGroup(len), repr_p, (prt)->view(prt,mask))
end


function getManifoldPartial(
    M::Manifolds.Circle,
    partial::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentation = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    mask = _checkManifoldPartialDims(M, partial, offset, doError)
    offset[] += manifold_dimension(M)
    return (M, repr, (prt)->view(prt,mask))
end

function getManifoldPartial(
    M::Manifolds.Rotations{TypeParameter{Tuple{2}}},
    partial::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentation = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    #
    mask = _checkManifoldPartialDims(M, partial, offset, doError)
    offset[] += manifold_dimension(M)
    return (M, repr, (prt)->view(prt,mask))
end

function getManifoldPartial(
    M::typeof(SpecialOrthogonalGroup(2)),
    partial::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentation = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    #
    mask = _checkManifoldPartialDims(M, partial, offset, doError)
    offset[] += manifold_dimension(M)
    return (M, repr, (prt)->view(prt,mask))
end

# near duplicate case for different repr ArrayPartition vs AbstractMatrix
function getManifoldPartial(
    M::typeof(SpecialEuclideanGroup(2; variant = :right)),
    partial::AbstractVector{Int},
    repr::Union{<:ArrayPartition, Nothing} = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    #FIXME This doesn't seem correct:
    # How is it used?
    # Is the partial dimension coupled or not?
    # A SE(2) prior will have different results than a product prior.
    if partial == [1, 2, 3]
        offset[] += manifold_dimension(M)
        return (M, repr, (s)->s)
    else
        return getManifoldPartial(
            ProductLieGroup(TranslationGroup(2), SpecialOrthogonalGroup(2)),
            partial,
            repr,
            offset;
            doError,
        )
    end
end
# near duplicate case for different repr ArrayPartition vs AbstractMatrix
function getManifoldPartial(
    M::typeof(LieGroups.SpecialEuclideanGroup(2; variant = :right)),
    partial_::Union{<:AbstractVector{Int}, <:Tuple},
    repr::AbstractMatrix = LinearAlgebra.I(3),
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    #
    # mask = _checkManifoldPartialDims(M, partial, offset, doError)
    offset[] += manifold_dimension(M)
    partial = _tuple(partial_)
    if partial == (1,)
        return (LieGroups.TranslationGroup(1), SVector(0.0,), (prt)->view(prt,1:1,3))
    elseif partial == (2,)
        return (LieGroups.TranslationGroup(1), SVector(0.0,), (prt)->view(prt,2:2,3))
    elseif partial == (1, 2)
        return (LieGroups.TranslationGroup(2), SVector(0.0, 0.0), (prt)->view(prt,1:2,3))
    elseif partial == (3,)
        return (LieGroups.SpecialOrthogonalGroup(2), repr, (prt)->view(prt,1:2,1:2))
    else
        error("SpecialEuclideanGroup(2) partial dimensions $partial not implemented yet")
    end
end


function getManifoldPartial(
    PrG::LieGroup{ℝ, <:ProductGroupOperation, <:ProductManifold},
    partial::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentationProduct = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    _checkManifoldPartialDims(PrG, partial, offset, doError)

    # loop through the ProductManifold components 
    ManiArr = []
    ReprArr = []
    lookups = []

    subgroups = map(LieGroup, PrG.manifold.manifolds, PrG.op.operations)
    for (i, m) in enumerate(subgroups)
        mask = _checkManifoldPartialDims(m, partial, offset, false)
        if any(mask)
            Mp, lkup = if repr === nothing
                # decide if representation should also be updated or left as nothing
                Mp, _, lkup = getManifoldPartial(m, partial, nothing, offset; doError = false)
                Mp, lkup
            else
                # hard assumption that repr::ArrayPartition to go along with M::ProductManifold
                # NOTE submanifold_component is the correct way to avoid this assumption
                Mp, Rp, lkup = getManifoldPartial(
                    m,
                    partial,
                    submanifold_component(PrG, repr, i),
                    offset;
                    doError = false,
                )
                push!(ReprArr, Rp)
                Mp, lkup
            end
            push!(ManiArr, Mp)
            push!(lookups, lkup)
        else
            offset[] += manifold_dimension(m)
            push!(lookups, (s) -> ())
        end
    end

    # trivial case, drop the ProductManifold for single element
    # if length(ManiArr) == 1
    #     repr_p = repr === nothing ? nothing : ReprArr[1]
    #     return (ManiArr[1], repr_p, (prt)->ArrayPartition(prt[1:1]))
    # elseif 1 < length(ManiArr)
        repr_p = repr === nothing ? nothing : ArrayPartition(ReprArr...)
        lookup = (point) -> begin
            elms = []
            for (j, pt) in enumerate(point.x)
                s = lookups[j](pt)
                if s !== ()
                    push!(elms, s)
                end
            end  
            ArrayPartition(elms...)
        end
        rettyp = if length(ManiArr) == 1
            ManiArr[1]
        else
            ProductLieGroup(ManiArr...)
        end
        return (rettyp, repr_p, lookup)
    # end
    return error("partial manifold calculations should not reach here")
end


"""
    $SIGNATURES

A so-called full dimension manifold can possibly be reduced to smaller partial manifolds over 
some of the dimensions, returning a new programatically generated `<:AbstractManifold`.
This function can optionally also reduce a point representation for the desired 
partial dimensions too.

Example
```julia
using Manifolds
using ApproxManifoldProducts

# a familiar manifold of translation and rotation in 2D
M = SpecialEuclideanGroup(2; variant = :right)


# make a new partial of only the translation components
M_x, _ = getManifoldPartial(M,[1;])
# returns a new TranslationGroup(1) corresponding to just x dimension

# representation is semidirect product of translation and rotation matrix
u0 = ArrayPartition([0.0;0],[1 0; 0 1.0])

# known coordinates are [x,y,θ], eg
#   vee(M,identity_element(M,u0),log(M,identity_element(M,u0),u0))
#   [0;0;0] in this example

# make another new partial of only the rotation component
M_rot, u_rot = getManifoldPartial(M,[3],u0)
# returns SpecialOrthogonalGroup(2) information

# make another new partial of only  y and θ
M_yθ, u_yθ = getManifoldPartial(M,[2;3],u0)
# returns new manifold information as ProductArray(TranslationGroup(1),SpecialOrthogonalGroup(2))
```

Notes
- Partial dimensions of interest are defined via a `AbstractVector{Int}` of coordinate dimensions.
- assumed to follow coordinates as transition step towards more general solution
- assume ProductManifold is the only way to stitch multiple manifolds together
- Still experimental.

DevNotes
- FIXME any semidirect product or action information is lost and naive Product manifold assumptions are currently made.

Related

[`getManifold`](@ref), [`AbstractManifold`], [`ProductManifold`], [`GroupManifold`], [`ArrayPartition`]
"""
function getManifoldPartial(
    M::ProductManifold,
    partial::Union{<:AbstractVector{Int}, <:Tuple},
    repr::_PartiableRepresentationProduct = nothing,
    offset::Base.RefValue{Int} = Ref(0);
    doError::Bool = true,
)
    #
    _checkManifoldPartialDims(M, partial, offset, doError)

    # loop through the ProductManifold components 
    ManiArr = []
    ReprArr = []
    for (i, m) in enumerate(M.manifolds)
        mask = _checkManifoldPartialDims(m, partial, offset, false)
        if any(mask)
            Mp = if repr === nothing
                # decide if representation should also be updated or left as nothing
                Mp,_ , = getManifoldPartial(m, partial, nothing, offset; doError = false)
                Mp
            else
                # hard assumption that repr::ArrayPartition to go along with M::ProductManifold
                # NOTE submanifold_component is the correct way to avoid this assumption
                Mp, Rp,  = getManifoldPartial(
                    m,
                    partial,
                    submanifold_component(repr, i),
                    offset;
                    doError = false,
                )
                push!(ReprArr, Rp)
                Mp
            end
            push!(ManiArr, Mp)
        else
            offset[] += manifold_dimension(m)
        end
    end

    # trivial case, drop the ProductManifold for single element
    if length(ManiArr) == 1
        repr_p = repr === nothing ? nothing : ReprArr[1]
        return (ManiArr[1], repr_p)
    elseif 1 < length(ManiArr)
        repr_p = repr === nothing ? nothing : ArrayPartition(ReprArr...)
        return (ProductManifold(ManiArr...), repr_p)
    end
    return error("partial manifold calculations should not reach here")
end



## ====================================================================
## POSSIBLE LEGACY FUNCTIONS BELOW, TODO REFACTOR OR DELETE
## ====================================================================



# partial (i.e. active) coordinate dimensions are left unchanged, while inactive 
# dimensions are set to default values (1.0 for variances, 0.0 for covariances)
_partialCovToDefault!(::Nothing, s) = s
function _partialCovToDefault!(p::Union{<:Tuple, <:AbstractVector{<:Integer}}, v::AbstractVector)
    mask = ones(Int, length(v)) .== 1
    mask[p] .= false
    v[mask] .= 1.0 # FIXME, = Inf instead
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





#
