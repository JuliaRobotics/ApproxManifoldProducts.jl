# Circular Manifold definition and associated arithmetic





struct Circular <: Manifold
  dof::Int
  addop::Function
  diffop::Function
  getMu
  getLambda
  domain::Tuple{Float64, Float64}
  Circular() = new(1,
                   addtheta,
                   difftheta,
                   getCircMu,
                   getCircLambda,
                   (-pi+0.0,pi-1e-15))
end




function get2DMu(mus, Lambdas; diffop::Function=-, periodicmanifold::Function=(x)->x, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5) )::Float64
  # TODO: can be solved as the null space basis, but requires proper scaling
  gg = (res, x) -> solveresid2DLinear!(res, x, mus, Lambdas, diffop=diffop)
  initr = initrange[2]-initrange[1]
  x0 = [initr*rand()+initrange[1]]
  r = NLsolve.nlsolve(gg, x0)
  xs = sign(r.zero[1] - x0[1])*1e-3 .+ r.zero
  r = NLsolve.nlsolve(gg, xs)
  return periodicmanifold(r.zero[1])
end


function get2DMuMin(mus, Lambdas; diffop::Function=-, periodicmanifold::Function=(x)->x, initrange::Tuple{Float64, Float64}=(-1e-5,1e-5), method=Optim.Newton(), Λμ::Bool=false )::Float64
  # TODO: can be solved as the null space basis, but requires proper scaling
  res = zeros(1)
  # @show round.(mus, digits=3)
  gg = (x) -> (solveresid2DLinear(res, x, mus, Lambdas, diffop=diffop))^2
  initr = initrange[2]-initrange[1]
  # TODO -- do we need periodicmanifold here?
  x0 = [initr*rand()+initrange[1]]
  r = Optim.optimize(gg, x0, method)
  # @show r.minimizer[1]
  return periodicmanifold(r.minimizer[1])
end





# struct SO2Manifold <: Manifold
# end
#
#
# # should not be defined in AMP, since we want IIF indepent of manifolds
# function *(PP::Vector{MB{SO2Manifold,B}}) where B
#   @info "taking manifold product of $(length(PP)) terms"
#   @warn "SO2Manifold: work in progress"
# end
#
# mbr1 = ManifoldBelief(SO2Manifold, 0.0)
# mbr2 = ManifoldBelief(SO2Manifold, 0.0)
#
# *([mbr1;mbr2])




#
