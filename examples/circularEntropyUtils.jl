# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold


global const reci_s2pi=1.0/sqrt(2.0*pi) # 1.0/2.5066282746310002

function normDistAccAt!(ret::AV,
                        idx::Int,
                        x::Float64,
                        sigma::Float64,
                        w::Float64=1.0  )::Nothing where {AV <: AbstractVector}
    global reci_s2pi
    @fastmath ret[idx] += w*reci_s2pi/sigma * exp( - (x^2)/(2.0*(sigma^2)) )
    return nothing
end

function rbfAccAt!(ret::AV,
                   idx::Int,
                   x::Float64,
                   μ::Float64=0.0,
                   σ::Float64=1.0,
                   w::Float64=1.0,
                   diffop::Function=-)::Nothing  where {AV <: AbstractVector}
    #
    normDistAccAt!(ret, idx, diffop(x, μ), σ, w)
    nothing
end
function rbf!(ret::AV,
              x::Float64,
              μ::Float64=0.0,
              σ::Float64=1.0   )::Nothing  where {AV <: AbstractVector}
    #
    ret[1] = 0.0
    normDistAccAt!(ret, 1, x-μ, σ)
    nothing
end


function rbf(x::Float64, μ::Float64=0.0, σ::Float64=1.0)
    ret = Vector{Float64}(undef, 1) # initialized in rbf!(..)
    rbf!(ret, x, μ, σ)
    return ret[1]
end




function looCrossValidation(pts::Array, bw::Float64; own=true)
    N = maximum(size(pts))
    reci_N = 1.0/(N-1)
    h = [bw;]
    loo = zeros(N)
    @inbounds for i in 1:N
        if !own
            # validation testing
            pts99 = pts[[1:(i-1);(i+1):end]]
            p99 = kde!(pts99, h)
            loo[i] = log(p99([pts[i];])[1])
        else
            # own naive entropy calculation
            loo[i] = 0.0
            for j in 1:N
                if i != j
                    normDistAccAt!(loo, i, pts[i]-pts[j], bw, reci_N)
                end
            end
            loo[i] = log(loo[i])
        end
    end
    return  sum(loo)/N
end





#
