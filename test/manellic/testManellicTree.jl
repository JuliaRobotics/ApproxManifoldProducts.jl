
using Test
using ApproxManifoldProducts
using StaticArrays
import ApproxManifoldProducts: HyperEllipse, ManellicTree, eigenCoords, splitPointsEigen

##

# test 
function testEigenCoords(
  r_C = pi/3,
  ax_CC = [SA[5*randn();randn()] for _ in 1:100],
)
  M = TranslationGroup(2)
  _R(α, s=exp(-α*im)) = real(s)*SA[1 0; 0 1] + imag(s)*SA[0 1; -1 0]
  # _R(α) = SA[cos(α) sin(α); -sin(α) cos(α)]
  r_R_ax = _R(r_C)
  # rotate coordinates
  r_CC = map(ax_CC) do ax_C
    r_R_ax*ax_C + SA[10;-100]
  end
  r_CV = Manifolds.cov(M, r_CC)
  r_R_ax_, L, pidx = eigenCoords(r_CV)

  # spot check
  @show _ax_ERR = log_lie(SpecialOrthogonal(2), (r_R_ax_')*r_R_ax)[1,2]
  @show testval = isapprox(0, _ax_ERR; atol = 5/length(ax_CC))
  @assert testval "Spot check failed on eigen split of manifold points, the estimated point rotation matrix did not match construction."

  r_CC, r_R_ax_, pidx, r_CV
end

##
@testset "test ManellicTree construction" begin
##

M = TranslationGroup(2)
r_CC, R, pidx, r_CV = testEigenCoords(pi/3);
ax_CCp, mask = splitPointsEigen(M, r_CC)
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

