# Circular Manifold definition and associated arithmetic




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







#
