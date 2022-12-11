
export  
  # new local features
  AMP,
  MKD,
  AbstractManifold,
  ManifoldKernelDensity,
  get2DLambda,
  get2DMu,
  get2DMuMin,
  resid2DLinear,
  solveresid2DLinear!,
  solveresid2DLinear,
  *,
  isapprox,

  # APi and util functions
  buildHybridManifoldCallbacks,
  getKDEManifoldBandwidths,
  manifoldProduct,
  manikde!,
  isPartial,
  calcProductGaussians

export getPoints, getBW, Ndim, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export calcMean
export mean, cov, std, var
export getInfoPerCoord, getBandwidth
export antimarginal
