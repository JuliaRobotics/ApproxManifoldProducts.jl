# NOTE none of this is exported, and follow upstreaming at JuliaRobotics/DistributedFactorGraphs.jl/issues/727
# copied from original in DFG/src/LightDFG/FactorGraphs/BiMaps.jl

# will be overloaded
import Base: getindex, setindex!, firstindex, lastindex, iterate, keys, isempty

struct _BiDictMap{T <: Integer}
    int_sym::Dict{T,Symbol}
    sym_int::Dict{Symbol,T}
end

_BiDictMap{T}(;sizehint=100) where T<:Integer = begin
    int_sym = Dict{T,Symbol}()
    sizehint!(int_sym, sizehint)
    sym_int = Dict{Symbol,T}()
    sizehint!(sym_int, sizehint)
    _BiDictMap{T}(int_sym, sym_int)
end

_BiDictMap(;sizehint=100) = _BiDictMap{Int}(;sizehint=sizehint)


Base.getindex(b::_BiDictMap, key::Int) = b.int_sym[key]
Base.getindex(b::_BiDictMap, key::Symbol) = b.sym_int[key]

# setindex!(b, value, key) = b[key] = value
function Base.setindex!(b::_BiDictMap, s::Symbol, i::Int)
    haskey(b.sym_int, s) && delete!(b.int_sym, b[s])
    haskey(b.int_sym, i) && delete!(b.sym_int, b[i])

    b.int_sym[i] = s
    b.sym_int[s] = i
end

function Base.setindex!(b::_BiDictMap, i::Int, s::Symbol)
    haskey(b.int_sym, i) && delete!(b.sym_int, b[i])
    haskey(b.sym_int, s) && delete!(b.int_sym, b[s])

    b.int_sym[i] = s
    b.sym_int[s] = i
end

function Base.delete!(b::_BiDictMap, i::Int)
    s = b[i]
    delete!(b.int_sym, i)
    delete!(b.sym_int, s)
    return b
end

Base.haskey(b::_BiDictMap, s::Symbol) = haskey(b.sym_int, s)
Base.haskey(b::_BiDictMap, i::Int) = haskey(b.int_sym, i)

Base.length(b::_BiDictMap) = length(b.int_sym)
#NOTE This will work only with LightGraphs that assumes indices 1:nv(g)
Base.firstindex(v::_BiDictMap) = 1
Base.lastindex(v::_BiDictMap) = length(v.int_sym)
Base.iterate(v::_BiDictMap, i=1) = (length(v.int_sym) < i ? nothing : (v.int_sym[i], i + 1))
Base.keys(v::_BiDictMap) = Base.OneTo(length(v.int_sym))
Base.isempty(v::_BiDictMap) = (length(v.int_sym) == 0)
