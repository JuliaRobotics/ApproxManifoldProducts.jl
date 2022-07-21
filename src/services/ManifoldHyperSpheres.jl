# The NearestNeighbors.jl package is licensed under the MIT "Expat" License:

# > Copyright (c) 2015: Kristoffer Carlsson.
# > Copyright (c) 2022: Dehann Fourie
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# const NormMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski,WeightedEuclidean,WeightedCityblock,WeightedMinkowski,Mahalanobis}


struct ManifoldHyperSphere{C <:AbstractArray, T <: AbstractFloat}
    """ center of the hypersphere, represented as a point on the manifold """
    center::C
    """ radius of the hypersphere over all dimensions.  TODO, upgrade to radius per product submanifold spaces. """ 
    r::T
end
# ManifoldHyperSphere(center::SVector{N,T1}, r::T2) where {N, T1, T2} = ManifoldHyperSphere(center, convert(T1, r))


@inline function NNR.intersects(m::DST.Metric,
                            s1::ManifoldHyperSphere,
                            s2::ManifoldHyperSphere)
    NNR.evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function NNR.encloses(m::DST.Metric,
                          s1::ManifoldHyperSphere,
                          s2::ManifoldHyperSphere)
    NNR.evaluate(m, s1.center, s2.center) + s1.r <= s2.r
end

# @inline function interpolate(::M,
#                              c1::V,
#                              c2::V,
#                              x,
#                              d,
#                              ab) where {V <: AbstractVector, M <: NormMetric}
#     alpha = x / d
#     @assert length(c1) == length(c2)
#     @inbounds for i in eachindex(ab.center)
#         ab.center[i] = (1 - alpha) .* c1[i] + alpha .* c2[i]
#     end
#     return ab.center, true
# end

@inline function NNR.interpolate(::M,
                            c1::V,
                            ::V,
                            ::Any,
                            ::Any,
                            ::Any ) where {V <: AbstractVector, M <: DST.Metric}
    return c1, false
end

function NNR.create_bsphere(
        mani::AbstractManifold, 
        data::AbstractVector{V}, 
        metric::DST.Metric, 
        indices::Vector{Int}, 
        low, 
        high, 
        ab
    ) where {V}
    #
    n_points = high - low + 1
    # First find center of all points
    mn = mean(mani, view(data, low:high))
    @info "THIS" typeof(mn) typeof(ab.center)
    _setc!(dst::AbstractVector, src::AbstractVector) = (dst[:] .= src[:])
    _setc!(dst::AbstractMatrix, src::AbstractMatrix) = (dst[:,:] .= src[:,:])

    _setc!(ab.center, mn)

    # Then find r
    r = zero(NNR.get_T(eltype(V)))
    for i in low:high
        r = max(r, NNR.evaluate(metric, data[indices[i]], ab.center))
    end
    r += eps(NNR.get_T(eltype(V)))
    return ManifoldHyperSphere(ab.center, r)
end

# Creates a bounding sphere from two other spheres
function NNR.create_bsphere(mani::AbstractManifold,
                        m::DST.Metric,
                        s1::ManifoldHyperSphere{C,T},
                        s2::ManifoldHyperSphere{C,T},
                        ab) where {C, T <: AbstractFloat}
    if NNR.encloses(m, s1, s2)
        return ManifoldHyperSphere(s2.center, s2.r)
    elseif NNR.encloses(m, s2, s1)
        return ManifoldHyperSphere(s1.center, s1.r)
    end

    # Compute the distance x along a geodesic from s1.center to s2.center
    # where the new center should be placed (note that 0 <= x <= d because
    # neither s1 nor s2 contains the other)
    dist = NNR.evaluate(m, s1.center, s2.center)
    x = 0.5 * (s2.r - s1.r + dist)
    center, is_exact_center = NNR.interpolate(m, s1.center, s2.center, x, dist, ab)
    if is_exact_center
        rad = 0.5 * (s2.r + s1.r + dist)
    else
        rad = max(s1.r + NNR.evaluate(m, s1.center, center), s2.r + evaluate(m, s2.center, center))
    end

    return ManifoldHyperSphere(center, rad)
end
