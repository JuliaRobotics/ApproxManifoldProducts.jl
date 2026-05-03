
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

# abstract type AbstractDensityRepresentation{partial, truncation} end
# const ReprType = AbstractDensityRepresentation

struct HomotopyRepresentation{
  statetype <: Union{<:AbstractManifold, <:DistributedFactorGraphs.AbstractStateType},
  partial, 
  reprtype, 
  truncation <: AbstractHomotopyTruncation
} end


struct MajorMaxDepth{
  N
} <: AbstractHomotopyTruncation end


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

getManifold(::HomotopyRepresentation{
  statetype, 
  partial, 
  reprtype, 
  truncation
}) where {statetype, partial, reprtype, truncation} = 
  getManifold(statetype)

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


