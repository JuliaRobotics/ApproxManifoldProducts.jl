# Euclidean Manifold definitions and arithmetic


struct Euclid2 <: Manifold
  dof::Int
  addop::Function
  diffop::Function
  getMu
  getLambda
  domain::Tuple{Tuple{Float64,Float64},Tuple{Float64, Float64}}
  Euclid2() = new(2, +, -, KDE.getEuclidMu, KDE.getEuclidLambda, ((-Inf,Inf),(-Inf,Inf)))
end

# ??
struct EuclideanManifold <: Manifold
end



get2DLambda(Lambdas::Vector{Float64})::Float64 = sum(Lambdas)



function *(PP::Vector{MB{EuclideanManifold,BallTreeDensity}})
  bds = Vector{BallTreeDensity}(undef, length(PP))
  for p in PP
    bds[i] = p.belief
  end
  *(bds)
end


#
