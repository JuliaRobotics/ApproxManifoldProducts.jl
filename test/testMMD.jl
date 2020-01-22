# test MMD distance

# testing only
# using Revise
# using Gadfly
# Gadfly.set_default_plot_size(35cm,25cm)

using Test
using ApproxManifoldProducts
using DataFrames

const AMP = ApproxManifoldProducts



@testset "Test mmd distance between Euclidean beliefs" begin

# when not plottin
offsets = 1:1 # -30:0.25:30
mc      = 1   # 1:100

checkGrid = zeros(length(offsets), length(mc))

df = DataFrame(offsets=[], mmd=[])


for i in 1:length(offsets), j in mc

P = randn(1,100)
Q = randn(1,100) .+ offsets[i]

res = zeros(1)

AMP.mmd!(res, P, Q, AMP.Euclid, bw=[0.001])

checkGrid[i,j] = res[1]
push!(df, (offsets[i],res[1]))

end

# when not plotting
@test 0 < df[!,:mmd][1] < 1.0

# plot(df, x=:offsets, y=:mmd, Geom.boxplot,
# Theme(default_color="MidnightBlue"))

# (manikde!(P, (:Euclid,)) |> getBW)[1,1]


end



@testset "Test mmd distance between 2D Euclidean beliefs" begin

# when not plottin
offsets = 1:1 # -30:0.25:30
mc      = 1   # 1:100

checkGrid = zeros(length(offsets), length(mc))

df = DataFrame(offsets=[], mmd=[])


for i in 1:length(offsets), j in mc

P = randn(2,100)
Q = randn(2,100) .+ offsets[i]

res = zeros(1)

AMP.mmd!(res, P, Q, AMP.Euclid2, bw=[0.001])

checkGrid[i,j] = res[1]
push!(df, (offsets[i],res[1]))

end

# when not plotting
@test 0 < df[!,:mmd][1] < 1.0

# plot(df, x=:offsets, y=:mmd, Geom.boxplot,
# Theme(default_color="MidnightBlue"))

# (manikde!(P, (:Euclid,)) |> getBW)[1,1]

end
