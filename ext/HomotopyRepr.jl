
using DistributedFactorGraphs

import DistributedFactorGraphs: getManifold


export 
  AbstractHomotopyTruncation,
  MajorMaxDepth
  # HomotopyRepresentation,

export 
  getStateType, 
  getManifold, 
  getReprType, 
  getTruncation, 
  getPartial


abstract type AbstractHomotopyTruncation end

getMajorsLength(::Type{<:AbstractHomotopyTruncation}) = 1

# abstract type AbstractDensityRepresentation{partial, truncation} end
# const ReprType = AbstractDensityRepresentation

struct HomotopyRepresentation{
  statetype <: Union{<:AbstractManifold, <:DistributedFactorGraphs.AbstractStateType},
  partial, 
  reprtype, 
  truncation <: AbstractHomotopyTruncation
} 
  _statekind::statetype
end

getMajorsLength(repr::HomotopyRepresentation) = getMajorsLength(getTruncation(repr))


struct MajorMaxDepth{
  N
} <: AbstractHomotopyTruncation end

# Binary tree with N major levels
getMajorsLength(::Type{MajorMaxDepth{N}}) where {N} = N^2 - 1



# trivial case -- should be in DFG instead FIXME
getManifold(manif::AbstractManifold) = manif



getStateType(::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  statetype

getPartial(::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  partial

getManifold(repr::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  getManifold(repr._statekind) # supports both DFG and ManifoldsBase

getReprType(::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  reprtype
                                
getTruncation(::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  truncation


