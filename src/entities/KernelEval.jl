
abstract type AbstractKernel end


## FIXME, export
# export DensityKernel, ConcentratedGaussianKernel

@kwdef struct DensityKernel{
    partial, # partial info for compiler, usually a value e.g. nothing or (1,3)
    K, # kernel info for compiler
    T  # additional parameters
} <: AbstractKernel
    """ Mixture/nonparametric weight value """
    weight::Float64 = 1.0
    """ functional basis such as RBF/MvNormal, Epanechnikov, (wavelet basis) etc. """
    functional::K = MvNormal(SMatrix{1,1}(1.0))
    """ additional parameters relating to on-manifold operations or similar """
    params::T = nothing
end

# helper constructor for common case without partials
DensityKernel{partial}(;
    weight::Float64,
    functional::K,
    params::T, 
    # partial::P = nothing,
) where {
    partial,
    K,
    T
} = DensityKernel{
    partial,
    K,
    T
}(;
    weight,
    functional,
    params
)

const ConcentratedGaussianKernel(;
    weight=1.0,
    p=SVector(0.0),  # center/expansion point on the manifold
    covmat=SMatrix{1,1}(1.0),
    partial::P = nothing,
) where P <: Union{Nothing, <:Tuple} = DensityKernel{partial}(;
    weight, 
    functional=MvNormal(covmat), # NOTE, find inverse Cholesky in MvNormal structure
    params=p,
)





## LEGACY BELOW

struct MvNormalKernel{T <: DensityKernel} <: AbstractKernel
    shim::T
end


import Base: getproperty


# function Base.getproperty(k::MvNormalKernel, f::Symbol)
#     if f === :sqrt_iΣ
#         # super slow and hacky, but only legacy.  WIP replacing
#         cov(k) |> inv |> sqrt
#     else
#         return getproperty(k.shim, f)
#     end
# end



# @kwdef struct MvNormalKernel{P, T, M, iM} <: AbstractKernel
#     """ On-manifold point representing center (mean) of the MvNormal distribution """
#     μ::P
#     """ Zero-mean normal distribution with covariance """
#     p::MvNormal{T, M}
#     # TDB might already be covered in p.Σ.chol but having issues with SymPD (not particular to this AMP repo)
#     """ Manually maintained square root concentration matrix for faster compute, TODO likely duplicate of existing Distrubtions.jl functionality. """
#     sqrt_iΣ::iM = sqrt(inv(cov(p)))
#     """ Nonparametric weight value """
#     weight::Float64 = 1.0
# end


function MvNormalKernel(
    μ::AbstractArray, 
    σ::AbstractArray, 
    weight::Real = 1.0;
    partial::Union{Nothing, <:Tuple} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
)
    @warn "MvNormalKernel is deprecated, use ConcentratedGaussianKernel instead, barr partial [maxlog=10]" maxlog=10
    _μ(s::AbstractArray, _p::Nothing, pf::Union{Nothing, <:Function}) = s
    _μ(s::AbstractVector, _p::Tuple, pf::Nothing) = begin
        # FIXME, cannot assume straight coordinate partial indexing works for all array{1}'s
        _s = _forcemutable(s)
        _s[setdiff(1:length(s), _p)] .= NaN
        return _s
    end
    _μ(s::AbstractMatrix, _p::Tuple, pf::Nothing) = begin
        # HACK BY ASSUMING CALLER SOLVED MATRIX CASE??? OR FUNCTION DISPATCH???
        _s = _forcemutable(s)
        # FIXME ON FIRE, this does not work for Matrices!!!!
        itr = setdiff(1:length(s), _p) 
        _s[itr, :] .= NaN
        _s[:, itr] .= NaN
        return _s
    end
    _μ(s::AbstractArray, _p::Tuple, pf::Function) = pf(s)
    c_(s::AbstractMatrix, _p::Nothing) = s
    c_(s::AbstractVector, _p::Nothing) = diagm(s)    
    c_(s::AbstractMatrix, _p::Tuple) = _partialCovToDefault!(_p,_forcemutable(s))
    c_(s::AbstractVector, _p::Tuple) = diagm(_partialCovToDefault!(_p,_forcemutable(s)))
    # TODO _forcestatic
    Σ = c_(σ, partial)
    _c = projectSymPosDef(Σ)
    return MvNormalKernel(
        ConcentratedGaussianKernel(;
            weight = float(weight),
            p = _μ(μ, partial, partl_cb),
            covmat = _c, # cov(MvNormal(_c)),
            partial,
        )
    )
end


MvNormalKernel(; μ, p::MvNormal, weight = 1.0, partial = nothing, kw...) = MvNormalKernel(μ, cov(p), weight; partial, kw...)


function convert(
    ::Type{MvNormalKernel{
        ApproxManifoldProducts.DensityKernel{
            L,
            MvNormal{F,P,Z},
            S
        }
    }},
    src::MvNormalKernel,
) where {L,F,P,Z,S}

    _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
    _sap(::Type{ArrayPartition{T,_S}}) where {T,_S} = _S
    _new(s) = S(s)
    _new(s::ArrayPartition{T,O}) where {T,O} = ArrayPartition(begin
        S_ = _sap(S)
        [S_.parameters[i](v) for (i,v) in enumerate(s.x)]
    end...)

    m = _new(src.shim.params)

    MvNormalKernel(
        m,
        _matType(P)(cov(src.shim.functional)),
        src.shim.weight;
        partial = L
    )
end


# function MvNormalKernel(
#     μ::AbstractArray, 
#     σ::AbstractArray, 
#     weight::Real = 1.0
# )
#     c_(s::AbstractMatrix) = s
#     c_(s::AbstractVector) = diagm(s)
#     Σ = c_(σ)
#     _c = projectSymPosDef(Σ)
#     p = MvNormal(_c)
#     # NOTE, TBD, why not sqrt(inv(p.Σ)), this had an issue seemingly internal to PDMat.chol which breaks an already forced SymPD matrix to again be not SymPD???
#     sqrt_iΣ = sqrt(inv(_c))
#     return MvNormalKernel(; μ, p, sqrt_iΣ, weight = float(weight))
# end



# case for identical types not requiring any conversions
function Base.convert(
    ::Type{T},
    src::T,
) where {T <: MvNormalKernel}
    return src
end


# # case for different types requiring conversion
# function Base.convert(
#     ::Type{MvNormalKernel{T}},
#     src::MvNormalKernel,
# ) where {T}
#     #
#     _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
#     μ = convert(P, src.μ) # P(src.μ)
#     p = MvNormal(_matType(M)(cov(src.p)))
#     # sqrt_iΣ = iM(src.sqrt_iΣ)
#     return MvNormalKernel(μ, p, src.weight)
# end