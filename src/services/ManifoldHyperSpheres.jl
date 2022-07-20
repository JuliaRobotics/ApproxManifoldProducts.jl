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


struct ManifoldHyperSphere{C <: ArrayPartition, T <: AbstractFloat}
    """ center of the hypersphere, represented as a point on the manifold """
    center::C
    """ radius of the hypersphere over all dimensions.  TODO, upgrade to radius per product submanifold spaces. """ 
    r::T
end
# ManifoldHyperSphere(center::SVector{N,T1}, r::T2) where {N, T1, T2} = ManifoldHyperSphere(center, convert(T1, r))


@inline function intersects(m::M,
                            s1::ManifoldHyperSphere{C,T},
                            s2::ManifoldHyperSphere{C,T}) where {T <: ArrayPartition, C, M <: DST.Metric}
    evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function encloses(m::M,
                          s1::ManifoldHyperSphere{C,T},
                          s2::ManifoldHyperSphere{C,T}) where {T <: ArrayPartition, C, M <: DST.Metric}
    evaluate(m, s1.center, s2.center) + s1.r <= s2.r
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

@inline function interpolate(::M,
                             c1::V,
                             ::V,
                             ::Any,
                             ::Any,
                             ::Any) where {V <: AbstractVector, M <: DST.Metric}
    return c1, false
end

function create_bsphere(mani::AbstractManifold, data::AbstractVector{V}, metric::DST.Metric, indices::Vector{Int}, low, high, ab) where {V}
    n_points = high - low + 1
    # First find center of all points
    fill!(ab.center, 0.0)
    ab.center[:] .= Identity(mani)
    for i in low:high
        ab.center[j] += data[indices[i]][j]
        # for j in 1:length(ab.center)
        #     FIXME  
        # end
    end
    ab.center .*= 1 / n_points

    # Then find r
    r = zero(get_T(eltype(V)))
    for i in low:high
        r = max(r, NN.evaluate(metric, data[indices[i]], ab.center))
    end
    r += eps(get_T(eltype(V)))
    return ManifoldHyperSphere(SVector{length(V),eltype(V)}(ab.center), r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere(mani::AbstractManifold,
                        m::DST.Metric,
                        s1::ManifoldHyperSphere{C,T},
                        s2::ManifoldHyperSphere{C,T},
                        ab) where {C, T <: AbstractFloat}
    if encloses(m, s1, s2)
        return ManifoldHyperSphere(s2.center, s2.r)
    elseif encloses(m, s2, s1)
        return ManifoldHyperSphere(s1.center, s1.r)
    end

    # Compute the distance x along a geodesic from s1.center to s2.center
    # where the new center should be placed (note that 0 <= x <= d because
    # neither s1 nor s2 contains the other)
    dist = evaluate(m, s1.center, s2.center)
    x = 0.5 * (s2.r - s1.r + dist)
    center, is_exact_center = interpolate(m, s1.center, s2.center, x, dist, ab)
    if is_exact_center
        rad = 0.5 * (s2.r + s1.r + dist)
    else
        rad = max(s1.r + evaluate(m, s1.center, center), s2.r + evaluate(m, s2.center, center))
    end

    return ManifoldHyperSphere(center, rad)
end
