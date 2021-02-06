# Euclidean Manifold definitions and arithmetic


struct Euclid2 <: MB.Manifold{MB.ℝ}
  dof::Int
  addop::Function
  diffop::Function
  getMu
  getLambda
  domain::Tuple{Tuple{Float64,Float64},Tuple{Float64, Float64}}
end

Euclid2() = Euclid2(2, +, -, KDE.getEuclidMu, KDE.getEuclidLambda, ((-Inf,Inf),(-Inf,Inf)))

# ??
struct EuclideanManifold <: MB.Manifold{MB.ℝ}
end



get2DLambda(Lambdas::AbstractVector{<:Real}) = sum(Lambdas)


function *(PP::Vector{MB_{EuclideanManifold,BallTreeDensity}})
  bds = (p->p.belief).(PP)
  *(bds)
end


#
