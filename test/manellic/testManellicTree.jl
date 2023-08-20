
using Test
using ApproxManifoldProducts
import ApproxManifoldProducts: HyperEllipse, ManellicTree, eigenCoords, testEigenCoords, splitPointsEigen


##
@testset "test ManellicTree construction" begin
##

r_CC, R, pidx, r_CV = testEigenCoords(-pi/2);
ax_CCp, mask  = splitPointsEigen(M, r_CC)
@test sum(mask) == (length(r_CC) ÷ 2)
# using GLMakie
# 
# fig = Figure()
# ax = Axis(fig[1,1])
# ptsl = ax_CCp[mask]
# ptsr = ax_CCp[xor.(mask,true)]
# plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
# plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
# ax = Axis(fig[2,1])
# ptsl = r_CC[mask]
# ptsr = r_CC[xor.(mask,true)]
# plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
# plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
# fig

##



##
end

