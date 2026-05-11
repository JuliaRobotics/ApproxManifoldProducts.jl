
abstract type AbstractKernel <: AbstractDensityForm end


struct ConcentratedGaussianKernel{
    partial, # partial info for compiler, usually a value e.g. nothing or (1,3)
    K <: Distributions.MvNormal, # kernel info for compiler
    T  # additional parameters
} <: AbstractKernel
    """ Mixture/nonparametric weight value """
    weight::Float64
    """ functional basis such as RBF/MvNormal, Epanechnikov, (wavelet basis) etc. """
    functional::K
    """ additional parameters relating to on-manifold operations or similar """
    params::T
end

function ConcentratedGaussianKernel(
    weight::Float64 = 1.0,
    functional::K = MvNormal(SMatrix{1,1}(1.0)),
    params::T = nothing;
    partial::Union{Nothing, <:Tuple} = nothing,
) where {K, T}
    ConcentratedGaussianKernel{
        partial,
        K,
        T,
    }(
        weight,
        functional,
        params,
    )
end

function ConcentratedGaussianKernel(
    μ::AbstractArray, 
    σ::AbstractArray, 
    weight::Real = 1.0;
    partial::Union{Nothing, <:Tuple} = nothing,
    partl_cb::Union{Nothing, <:Function} = nothing,
)
    # @warn "MvNormalKernel is deprecated, use ConcentratedGaussianKernel instead, barr partial [maxlog=10]" maxlog=10
    _μ(s::AbstractArray, _p::Nothing, pf::Union{Nothing, <:Function}) = s
    _μ(s::AbstractArray, _p::Tuple, pf::Function) = begin
        _s = deepcopy(s)
        _s_ = pf(_s)
        fill!(s, NaN)
        s_ = pf(s) # required for non-trivial points, eg SO/SE have more complicated representations
        s_ .= _s_ # copy back only the partials, leave NaNs in the rest
        return s
    end

    c_(s::AbstractMatrix, _p::Nothing) = s
    c_(s::AbstractVector, _p::Nothing) = diagm(s)    
    c_(s::AbstractMatrix, _p::Tuple) = _partialCovToDefault!(_p,_forcemutable(s))
    c_(s::AbstractVector, _p::Tuple) = diagm(_partialCovToDefault!(_p,_forcemutable(s)))

    # TODO _forcestatic
    Σ = c_(σ, partial)
    _c = projectSymPosDef(Σ)
    functional = MvNormal(_c)
    params = _μ(μ, partial, partl_cb)
    return ConcentratedGaussianKernel(
        float(weight),
        functional,
        params,
    )
end


# ConcentratedGaussianKernel(; μ, p::MvNormal, weight = 1.0, partial = nothing, kw...) = ConcentratedGaussianKernel(μ, cov(p), weight; partial, kw...)


function ConcentratedGaussianKernel{
    L,
    MvNormal{F,P,Z},
    S
}(
    src::ConcentratedGaussianKernel;
    partl_cb::Union{Nothing, <:Function} = nothing, # partial is pulled from kernel...
    Σ = nothing,
) where {L,F,P,Z,S}

    _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
    _sap(::Type{ArrayPartition{T,_S}}) where {T,_S} = _S
    _new(s) = S(s)
    _new(s::ArrayPartition{T,O}) where {T,O} = ArrayPartition(begin
        S_ = _sap(S)
        [S_.parameters[i](v) for (i,v) in enumerate(s.x)]
    end...)

    m = _new(src.params)
    Σ_ = isnothing(Σ) ? cov(src.functional) : Σ

    ConcentratedGaussianKernel(
        m,
        _matType(P)(Σ_),
        src.weight;
        partial = L,
        partl_cb,
    )
end


function ConcentratedGaussianKernel(
    src::ConcentratedGaussianKernel{
        L,
        MvNormal{F,P,Z},
        S
    };
    partl_cb::Union{Nothing, <:Function} = nothing, # partial is pulled from kernel...
    Σ = nothing,
    kw...
) where {L,F,P,Z,S}
    return ConcentratedGaussianKernel{L,MvNormal{F,P,Z},S}(src; partl_cb, Σ, kw...)
end


