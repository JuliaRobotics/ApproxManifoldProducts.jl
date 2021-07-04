
## ================================================================================================================================
# Serialization
## ================================================================================================================================

# abstract type JSONManifoldKernelDensity end

# export JSONManifoldKernelDensity

function Base.convert(::Type{<:AbstractString}, 
                      mkd::ManifoldKernelDensity)
  #
  dict = Dict{Symbol, String}()
  dict[:_type] = "ManifoldKernelDensity"
  dict[:belief] = KDE.string( mkd.belief )
  dict[:manifold] = string(mkd.manifold)

  JSON2.write(dict)
end

function Base.convert(::Type{<:ManifoldKernelDensity}, str::AbstractString)
  dict = JSON2.read(str)
  # make module specific
  # good references: 
  #  https://discourse.julialang.org/t/converting-string-to-datatype-with-meta-parse/33024/2
  #  https://discourse.julialang.org/t/is-there-a-way-to-import-modules-with-a-string/15723/6
  manisplit = split(dict[:manifold], '.')
  manimod, manitp = if 1 < length(manisplit)
    modex = Symbol(manisplit[1])
    @eval($modex), dict[:manifold][(length(manisplit[1])+2):end]
  else
    Main, dict[:manifold]
  end
  manip = Meta.parse(manitp)
  manis = Core.eval(manimod, manip) # could not get @eval to work with $
  belief_ = convert(BallTreeDensity, dict[:belief])
  ManifoldKernelDensity(manis, belief_)
end


#