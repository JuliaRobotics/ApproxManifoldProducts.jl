module ApproxManifoldProducts

using NLsolve

export
  get2DLambda,
  get2DMu,
  resid2DLinear,
  solveresid2DLinear


get2DLambda(Lambdas) = sum(Lambdas)

function resid2DLinear(μ, mus, Lambdas)
dμ = mus .- μ
ret = sum( Lambdas.*dμ )
return ret
end

function solveresid2DLinear(res, x, mus, Lambdas)
res[1] = resid2DLinear(x, mus, Lambdas)
end

function get2DMu(mus, Lambdas)
  # TODO: can be solved as the null space basis, but requires proper scaling
  gg = (res, x) -> solveresid2DLinear(res, x, mus, Lambdas)
  r = NLsolve.nlsolve(gg, [0.0])
  return r.zero
end


end
