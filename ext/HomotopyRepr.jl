
# using DistributedFactorGraphs

# import DistributedFactorGraphs: getManifold

# DO NOT EXPORT AbstractPartialLegacyCompat 
const AbstractPartialLegacyCompat = Union{<:DistributedFactorGraphs.AbstractPartialTraits, Nothing, Tuple, Vector{Int}} 
# FIXME, drop Nothing rewire using empty tuple during refactor compat period

