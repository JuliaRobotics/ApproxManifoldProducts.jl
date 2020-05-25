
using ApproxManifoldProducts

# using JLD2
# @load "mmdse2.jld2" pred meas

pred = randn(3,100)
meas = randn(3,100)

val = [0.0]
mmd!(val, pred, meas, SE2_Manifold)
