# Interface

import Base: replace
export makeCoordsFromPoint, makePointFromCoords, getNumberCoords
export identity_element
export setPointPartial!, setPointsMani!
export replace

"""
    $SIGNATURES

Helper function to convert coordinates to a desired on-manifold point.

DevNotes
- FIXME need much better consolidation or even removal of this function entirely.
  - This function is only implemented on Lie groups

Notes
- `u0` is used to identify the data type for a point
- Pass in a different `exp` if needed.
"""
function makePointFromCoords(
    G::AbstractLieGroup,
    coords::AbstractVector{<:Real},
    u0 = zeros(manifold_dimension(G)),
)
    X = hat(LieAlgebra(G), coords, typeof(u0))
    return exp(G, X)
end

function makePointFromCoords(
    G::SpecialEuclideanGroup,
    coords::AbstractVector{<:Real},
    u0 = zeros(manifold_dimension(G)),
)
    X = hat(LieAlgebra(G), coords, typeof(u0))
    ε = identity_element(G, typeof(u0))
    #TODO - review - Force TR-coordinates on SE(n)
    return exp(base_manifold(G), ε, X)
end

# should perhaps just be dispatched for <:AbstractGroupManifold
# only works for AbstractGroupManifold (have an identity)

function makeCoordsFromPoint(G::AbstractLieGroup, pt)
    return vee(LieAlgebra(G), log(G, pt))
end

function makeCoordsFromPoint(G::SpecialEuclideanGroup, pt)
    #TODO - review - Force TR-coordinates on SE(n)
    p = ArrayPartition(ManifoldsBase.submanifold_components(G, pt))
    ϵ = identity_element(G, typeof(p))
    X = log(base_manifold(G), ϵ, p)
    return vee(LieAlgebra(G), X)
end

# Sphere(2) has 3 coords, even though the manifolds only has 2 dimensions (degrees of freedom)
getNumberCoords(M::MB.AbstractManifold, p) = length(makeCoordsFromPoint(M, p))

# TODO DEPRECATE
# related _pointsToMatrixCoords
function _matrixCoordsToPoints(M::MB.AbstractManifold, pts::AbstractMatrix{<:Real}, u0)
    #
    # ptsArr = Vector{Vector{Float64}}(undef, size(pts, 2))
    # @cast ptsArr[j][i] = pts[i,j]
    vecP = Vector{typeof(u0)}(undef, size(pts, 2))
    for j = 1:size(pts, 2)
        pt = pts[:, j]
        vecP[j] = makePointFromCoords(M, pt, u0)
    end
    return vecP
end

function _pointsToMatrixCoords(M::MB.AbstractManifold, pts::AbstractVector{P}) where {P}
    mat = zeros(manifold_dimension(M), length(pts))
    ϵ = identity_element(M, typeof(pts[1]))
    for (j, pt) in enumerate(pts)
        mat[:, j] = vee(M, ϵ, log(M, ϵ, pt))
    end

    return mat
end

# asPartial=true indicates that src coords are smaller than dest coords, and false implying src has dummy values in placeholder dimensions
function setPointPartial!(
    Mdest::AbstractManifold,
    dest,
    Msrc::AbstractManifold,
    src,
    partial::AbstractVector{<:Integer},
    asPartial::Bool = true,
)
    #
    # trivial case of empty factor
    if length(partial) == 0
        return dest
    end

    dest_ = AMP.makeCoordsFromPoint(Mdest, dest)
    # e0 = identity_element(Mdest, dest)
    # dest_ = vee(Mdest, e0, log(Mdest, e0, dest))

    # Note on partial cases.
    #  Mdest is always full dimensional as the destination of some new values.
    #  Msrc is partial dimension manifold.
    #  src is assumed to be values which only represent the partial values 

    # FIXME, does this line need to cater for both partial and tangent or point cases?
    src_ = AMP.makeCoordsFromPoint(Msrc, src)
    # e0s = identity_element(Msrc, src)
    # src_ = vee(Msrc, e0s, log(Msrc, e0s, src))

    # do the copy in coords 
    dest_[partial] .= asPartial ? src_ : view(src_, partial)

    # update points base in original
    dest__ = makePointFromCoords(Mdest, dest_, dest)
    # dest__ = exp(Mdest, e0, hat(Mdest, e0, dest_))
    setPointsMani!(dest, dest__)

    #
    return dest
end

function setPointPartial!(
    Mdest::AbstractManifold,
    dest::AbstractArray{T},
    Msrc::AbstractManifold,
    src::AbstractArray{U},
    partial::AbstractVector{<:Integer},
    destIdx,
    srcIdx = destIdx,
    asPartial::Bool = true,
) where {T <: AbstractArray, U <: AbstractArray}
    if isbitstype(T)
        #TODO needs cleanup, this is copied from setPointPartial! above with index changes
        if length(partial) == 0
            return dest[destIdx]
        end
        dest_coords = collect(AMP.makeCoordsFromPoint(Mdest, dest[destIdx]))
        src_coords = AMP.makeCoordsFromPoint(Msrc, src[srcIdx])
        dest_coords[partial] .= asPartial ? src_coords : view(src_coords, partial)
        return dest[destIdx] = makePointFromCoords(Mdest, dest_coords, dest[destIdx])

    else
        return setPointPartial!(Mdest, dest[destIdx], Msrc, src[srcIdx], partial, asPartial)
    end
end

#TODO workaround for supporting bitstypes, need rewrite, can consider `PowerManifoldNestedReplacing` or similar, maybe copyto!
function setPointsMani!(
    dest::AbstractArray{T},
    src::AbstractArray{U},
    destIdx,
    srcIdx = destIdx,
) where {T <: AbstractArray, U <: AbstractArray}
    if isbitstype(T) || T <: AbstractArray{<:Number, 0}
        dest[destIdx] = src[srcIdx]
    else
        setPointsMani!(dest[destIdx], src[srcIdx])
    end
end

function setPointsMani!(
    dest::AbstractArray{T},
    src::AbstractArray{T},
    destIdx,
    srcIdx = destIdx,
) where {T <: Array{<:Number, 0}}
    return dest[destIdx] = src[srcIdx]
end

function setPointsMani!(
    dest::AbstractArray{T},
    src::AbstractArray{U},
    destIdx,
) where {T <: AbstractArray, U <: Number}
    if isbitstype(T)
        dest[destIdx] = src
    elseif T <: AbstractArray{<:Number, 0}
        dest[destIdx] = fill(src[1])
    else
        setPointsMani!(dest[destIdx], src)
    end
end

#TODO  ArrayPartition should work for now as it's an AbstractVector, but it won't remain mutable
setPointsMani!(dest::AbstractVector, src::AbstractVector) = (dest .= src)
setPointsMani!(dest::AbstractMatrix, src::AbstractMatrix) = (dest .= src)
function setPointsMani!(dest::AbstractVector, src::AbstractMatrix)
    @assert size(src, 2) == 1 "Workaround setPointsMani! currently only allows size(::Matrix, 2) == 1"
    return setPointsMani!(dest, src[:])
end
function setPointsMani!(dest::AbstractMatrix, src::AbstractVector)
    @assert size(dest, 2) == 1 "Workaround setPointsMani! currently only allows size(::Matrix, 2) == 1"
    return setPointsMani!(view(dest, :, 1), src)
end

function setPointsMani!(dest::AbstractVector, src::AbstractVector{<:AbstractVector})
    @assert length(src) == 1 "Workaround setPointsMani! currently only allows Vector{Vector{P}}(...) |> length == 1"
    return setPointsMani!(dest, src[1])
end



#
