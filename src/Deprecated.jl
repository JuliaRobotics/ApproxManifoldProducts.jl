

## ======================================================================================================
## Remove below before v0.16
## ======================================================================================================



## ======================================================================================================
## Remove below before v0.15
## ======================================================================================================


@deprecate ManifoldKernelDensity(
  manifold::AbstractManifold,
  hode::HomotopyDensity,
  _partial::Union{Nothing, <:AbstractVector, <:Tuple},
  _u0::AbstractVector,
  infoPerCoord::AbstractVector{<:Real};
  partl_cb::Union{Nothing, <:Function} = nothing
) hode

# remove -- find workaround for partl_cb in this case
@deprecate ManifoldKernelDensity(
    hode::HomotopyDensity,
    ::Nothing;
    partl_cb::Nothing = nothing,
) hode


function ManifoldKernelDensity(
    bel::B,
    pl_mask::Union{<:BitVector, <:AbstractVector{<:Bool}},
) where {B <: HomotopyDensity}
    error("This ManifoldKernelDensity constructor is obsolete, use HomotopyDensity directly")
    # return bel
end


# function plotCircBeliefs(
#     arr::V;
#     N::Int = 1000,
#     th = range(-pi; stop = pi - 1e-15, length = N),
#     c = ["green"; "blue"; "deepskyblue"; "magenta"; "cyan"],
#     logpdf::Bool = true,
#     rVo::Vector{Float64} = [0.0; 0.0; 0.0],
#     radix::Float64 = 1.0,
#     text::String = "",
#     title::String = "",
#     legend = nothing,
# ) where {V <: Vector}
#     #
#     c = ["black"; c]
#     beliefs = Dict{Int, Function}()
#     beliefs[1] = (x) -> 1.0
#     j = 1
#     # TODO should be passing in addop, and diffop
#     for ar in arr
#         j += 1
#         beliefs[j] = logpdf ? (x) -> log(ar(x) + 1.0) : (x) -> ar(x) #
#     end

#     M = SpecialOrthogonalGroup(2)
#     e0 = Identity(M)

#     PL = []
#     # TODO loosing some modes in plotting???
#     for j = 1:length(beliefs)
#         obj = zeros(N)
#         X = zeros(N)
#         Y = zeros(N)
#         for i = 1:N
#             t = th[i]
#             obj[i] = radix * beliefs[1](t) # = 1.0
#             obj[i] += (j != 1 ? beliefs[j](t) : 0.0)
#             # second term directional statistics for plotting (not evaluation)
#             obj[i] += (j != 1 ? beliefs[j](t - 2pi) : 0.0)
#             obj[i] += (j != 1 ? beliefs[j](t + 2pi) : 0.0)
#             xy = exp(M, e0, hat(M, e0, rVo[3])) * [cos(t) * obj[i]; sin(t) * obj[i]]
#             X[i] = xy[1] + rVo[1]
#             Y[i] = xy[2] + rVo[2]
#         end
#         push!(
#             PL,
#             Gadfly.layer(;
#                 x = deepcopy(X),
#                 y = deepcopy(Y),
#                 label = [text;],
#                 Gadfly.Geom.path,
#                 Gadfly.Theme(; default_color = parse(Colorant, c[j])),
#                 Geom.label,
#             )[1],
#         )
#     end

#     push!(PL, Coord.cartesian(; aspect_ratio = 1.0))
#     push!(PL, Guide.title(title))
#     if legend != nothing
#         push!(PL, Guide.manual_color_key("Legend", legend, c[2:end]))
#     end

#     return Gadfly.plot(PL...)
# end

# function plotKDECircular(
#     bds::AbstractVector{<:BallTreeDensity};
#     c = ["green"; "blue"; "deepskyblue"; "magenta"; "cyan"],
#     logpdf::Bool = true,
#     scale::Real = 0.2,
#     offset::Real = 0.0,
#     rVo::AbstractVector{<:Real} = [0.0; 0.0; 0.0],
#     radix::Real = 1.0,
#     text::String = "",
#     title::String = "",
#     legend = nothing,
# )
#     #
#     arr = []

#     for bd in bds
#         gg = (x::Float64) -> scale * bd([x;], false, 1e-3, (addtheta,), (difftheta,))[1]
#         push!(arr, gg)
#     end

#     return plotCircBeliefs(
#         arr;
#         c = c,
#         logpdf = logpdf,
#         rVo = rVo,
#         radix = radix,
#         text = text,
#         title = title,
#         legend = legend,
#     )
# end

# function plotKDECircular(
#     bd::BallTreeDensity;
#     c = ["green";],
#     logpdf::Bool = true,
#     scale::Float64 = 0.2,
#     rVo::Vector{Float64} = [0.0; 0.0; 0.0],
#     radix::Float64 = 1.0,
#     text::String = "",
#     title::String = "",
#     legend = nothing,
# )
#     #
#     return plotKDECircular(
#         [bd;];
#         c = c,
#         logpdf = logpdf,
#         scale = scale,
#         rVo = rVo,
#         radix = radix,
#         text = text,
#         title = title,
#         legend = legend,
#     )
# end

# plotMKD(mkd::HomotopyDensity) = plotKDE(mkd)
# plotMKD(mkds::AbstractVector{<:HomotopyDensity}) = plotKDE(mkds)



# getPartial(x::ManifoldKernelDensity) = getPartial(x.shim)
# Ndim(x::ManifoldKernelDensity, w...; kw...) = Ndim(x.shim, w...; kw...)
# Npts(x::ManifoldKernelDensity, w...; kw...) = Npts(x.shim, w...; kw...)
# getWeights(x::ManifoldKernelDensity, w...; kw...) = getWeights(x.shim, w...; kw...)
# (x::ManifoldKernelDensity)(w...; kw...) = x.shim(w...; kw...)
# getPointRepr(x::ManifoldKernelDensity) = getPointRepr(x.shim)
# getManifold(x::ManifoldKernelDensity) = getManifold(x.shim)


# Base.length(::HomotopyDensity{L, M, D, N}) where {L, M, D, N} = N


# function Base.show(io::IO, hode::HomotopyDensity{partial,M}) where {partial,M}
#     _round(s::AbstractArray; kw...) = round.(s[:]; kw...)
#     _round(s::AbstractVector{<:AbstractMatrix}; kw...) = round.(s[1][:]; kw...)

#     printstyled(io, "HomotopyDensity{"; bold = true, color = :blue)
#     println(io)
#     # FIXME restore after HomotopyDensity refactor
#     printstyled(io, "    partial"; bold = true, color = :magenta)
#     print(io, " = ", partial, ",")
#     println(io)
#     printstyled(io, "    M"; bold = true, color = :magenta)
#     print(io, " = ", M, ",")
#     println(io)
#     # printstyled(io, "    B"; bold = true, color = :magenta)
#     # print(io, " = ", B, ",")
#     # println(io)
#     # FIXME restore after HomotopyDensity refactor
#     println(io, " }(")
#     println(io, "  Npts:  ", Npts(hode))
#     print(io, "  dims:  ", Ndim(hode))
#     printstyled(io, isPartial(hode) ? "* --> $(length(getPartial(hode)))" : ""; bold = true)
#     println(io)
#     println(io, "  prtl:   ", getPartial(hode))
#     bw = (getBW(hode).^2)[:, 1]
#     pvec = isPartial(hode) ? getPartial(hode) : collect(1:length(bw))
#     println(io, "  bws:   ", getBandwidth(hode, true) |> x -> _round(x; digits = 4)) # .|> x->round(x,digits=4))
#     println(io, "  ipc:   ", getInfoPerCoord(hode, true) .|> x -> round(x; digits = 4))
#     print(io, "   mean: ")
#     try
#         mn = mean(hode)
#         if mn isa ProductRepr # TODO UPDATE to ArrayPartition only, discontinued use of ProductRepr long ago.
#             println(io)
#             for prt in mn.parts
#                 println(io, "         ", round.(prt, digits = 4))
#             end
#         else
#             println(io, round.(mn', digits = 4))
#         end
#     catch
#         println(io, "----")
#     end
#     println(io, ")")
#     return nothing
# end
# Base.show(io::IO, ::MIME"text/plain", mkd::HomotopyDensity) = show(io, mkd)


# getPoints(mt::HomotopyDensity; permute::Bool = true) = permute ? view(mt.data, mt.permute) : mt.data

# """
#     $SIGNATURES

# Return underlying points used to construct the [`ManifoldKernelDensity`](@ref).

# Notes
# - Return type is `::Vector{P}` where `P` represents a Manifold point type (e.g. group element or coordinates).
# - Second argument controls whether partial dimensions only should be returned (`=true` default).

# DevNotes
# - Currently converts down to manifold from matrix of coordinates (legacy), to be deprecated TODO
# """
# function getPoints(
#     x::ManifoldKernelDensity,
#     aspartial::Bool = true;
#     permute::Bool = true,
# )
#     #
#     pts = getPoints(x.shim; permute)

#     if !isPartial(x) && !aspartial
#         error("MKD getPoints aspartial=true but MKD is not partial")
#         return pts
#     end

#     Mp, Rp, lkup = getManifoldPartial(getManifold(x), getPartial(x), pts[1])

#     vecP = Vector{typeof(Rp)}(undef, length(pts))
#     for (j,pt) in enumerate(pts)
#         vecP[j] =  lkup(pt)
#     end
#     return vecP
# end



# function getBW(
#     x::ManifoldKernelDensity{B},
#     asPartial::Bool = true;
#     kw...,
# ) where {B}
#     bws = getBW(x.shim; kw...)
#     if isPartial(x) && asPartial
#         return (bw->view(bw, getPartial(x))).(bws)
#     end
#     return bws
# end

# """
#     $TYPEDEF

# On-manifold kernel density belief.

# Notes
# - Allows partials as identified by list of coordinate dimensions e.g. `partial = [1;3]`
#   - When building a partial belief, use full points with necessary information in the specified partial coords.

# DevNotes
# - WIP AMP issue 41, use generic retractions during manifold products.
# """
# struct ManifoldKernelDensity{B <: HomotopyDensity}
#   # manifold::M
#   """ HomotopyDensity legacy-shim for hybrid-(non)parametric belief propagation """
#   shim::B
#   # _partial::L
#   # """ just an example point for local access to the point data type"""
#   # _u0::P
#   # infoPerCoord::Vector{Float64}
# end


# import Base: getproperty

# function getproperty(mkd::ManifoldKernelDensity, f::Symbol)
#   if f === :shim || f === :belief
#     return getfield(mkd, :shim)
#   elseif f === :manifold
#     return getManifold(mkd)
#   elseif f === :_partial
#     return getPartial(mkd)
#   elseif f === :_u0
#     return getPointRepr(mkd)
#   else
#     @warn "ManifoldKernelDensity has been deprecated, use HomotopyBelief instead." maxlog=20
#     getproperty(getfield(mkd, :shim), f)
#   end
# end

