


getTopologyKind(reprkind::HomotopyRepr) = reprkind.topologykind
getTopologyKind(hode::HomotopyDensity) = getTopologyKind(hode.reprkind)

getFormKind(repr::HomotopyRepr) = repr.formkind
getFormKind(hode::HomotopyDensity) = getFormKind(hode.reprkind)

getStateKind(repr::HomotopyRepr) = repr.statekind
getStateKind(hode::HomotopyDensity) = getStateKind(hode.reprkind)

getManifold(reprkind::HomotopyRepr) = getManifold(reprkind.statekind)
# getManifold(reprkind::HomotopyReprDFG) = getManifold(reprkind.statekind)
# getManifold(hode::HomotopyDensity) = getManifold(hode.reprkind)
# getManifold(state::State) = getManifold(state.belief)

getPointType(x::HomotopyDensity) = eltype(x.points) # TODO use HomotopyDensity{T} style instead
function getManifold(hode::HomotopyDensity, aspartial::Bool = false)
    return if !aspartial
        getManifold(hode.reprkind)
    else
        M_, _, _ = getManifoldPartial(getManifold(hode), getPartial(hode), hode.points[1])
        M_
    end
end


getPartial(repr::HomotopyRepr) = repr.partial
getPartial(hode::HomotopyDensity) = getPartial(hode.reprkind)

_vanillareprT(::T) where {T <:ConcentratedGaussianKernel} = ConcentratedGaussianKernel

# Binary tree with N major levels
getMajorsLength(::Type{BinaryTruncFixedDepth{N}}) where {N} = N^2 - 1
getMajorsLength(kind::AbstractHomotopyTopology) = getMajorsLength(typeof(kind))
getMajorsLength(repr::HomotopyRepr) = getMajorsLength(repr.topologykind)

function DistributedFactorGraphs.getDimension(
    st::Union{<:AbstractManifold, <:StateType}
)
    return manifold_dimension(getManifold(st))
end

function DistributedFactorGraphs.getDimension(
    hode::HomotopyDensity
)
    return getDimension(getStateKind(hode))
end




