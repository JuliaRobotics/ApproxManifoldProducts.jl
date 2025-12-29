
abstract type AbstractKernel end


## FIXME, export
# export HomotopyBeliefKernel, ConcentratedGaussianKernel

@kwdef struct HomotopyBeliefKernel{K, T} <: AbstractKernel
    """ Mixture/nonparametric weight value """
    weight::Float64 = 1.0
    """ functional basis such as RBF/MvNormal, Epanechnikov, (wavelet basis) etc. """
    functional::K = MvNormal(SMatrix{1,1}(1.0))
    """ additional parameters relating to on-manifold operations or similar """
    params::T = nothing
end

const ConcentratedGaussianKernel(;
    weight=1.0,
    p=SVector(0.0),
    devmat=SMatrix{1,1}(1.0)
) = HomotopyBeliefKernel(;
    weight, 
    functional=MvNormal(p, devmat^2), # NOTE, find inverse Cholesky in MvNormal structure
    params=(; p),
)





## LEGACY BELOW

struct MvNormalKernel{T <: HomotopyBeliefKernel} <: AbstractKernel
    shim::T
end


import Base: getproperty


function Base.getproperty(k::MvNormalKernel, f::Symbol)
    if f === :cov
        return cov(k.shim.functional)
    elseif f === :sqrt_iΣ
        # super slow and hacky, but only legacy.  WIP replacing
        cov(k) |> inv |> sqrt
    else
        return getproperty(k.shim, f)
    end
end



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
    weight::Real = 1.0
)
    c_(s::AbstractMatrix) = s
    c_(s::AbstractVector) = diagm(s)
    Σ = c_(σ)
    _c = projectSymPosDef(Σ)
    functional = MvNormal(_c)
    MvNormalKernel(
        ConcentratedGaussianKernel(
            weight = float(weight),
            p = μ,
            devmat = sqrt(cov(functional)),
        )
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

# case for different types requiring conversion
function Base.convert(
    ::Type{MvNormalKernel{T}},
    src::MvNormalKernel,
) where {T}
    #
    _matType(::Type{Distributions.PDMats.PDMat{_F, _M}}) where {_F, _M} = _M
    μ = convert(P, src.μ) # P(src.μ)
    p = MvNormal(_matType(M)(cov(src.p)))
    # sqrt_iΣ = iM(src.sqrt_iΣ)
    return MvNormalKernel(μ, p, src.weight)
end