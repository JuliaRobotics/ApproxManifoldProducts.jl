
abstract type AbstractKernel end


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

