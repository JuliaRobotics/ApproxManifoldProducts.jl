
using ApproxManifoldProducts
using Test


##

function directProductGaussianTestHelper(
    M,
    P1::HomotopyDensity,
    P2::HomotopyDensity,
    P12::HomotopyDensity,
    sl::Vector{Vector{Int}},
    pts1::Vector{Vector{Float64}},
    pts2::Vector{Vector{Float64}},
    N::Int,
)

    invpermute(B::HomotopyDensity, s::Int) = findfirst(==(s), B.permute)
    # use idx 1 assuming all leaf bandwidths are the same
    bw1 = getBW(P1)[invpermute(P1,1)] .^ 2
    bw2 = getBW(P2)[invpermute(P2,1)] .^ 2

    sl1 = [s[1] for s in sl]
    sl2 = [s[2] for s in sl]
    # layers and layers of belief tree indexing pain (part of refactoring transition for HomotopyDensity rename)
    sl1_ = sl1[1] % N
    sl1_ = sl1_ == 0 ? N : sl1_
    sl2_ = sl2[1] % N
    sl2_ = sl2_ == 0 ? N : sl2_
    
    uhm = ApproxManifoldProducts.calcProductKernelsBTLabels(
      M,
      [P1; P2],
      [(sl1[1],sl2[1]);],
      false;
    )
      
    u1 = pts1[sl1_]
    u2 = pts2[sl2_]
    # @info "beforeprod" u1 u2 bw1 bw2
    u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
    @test isapprox(mean(uhm[1]), u12)

    pts12 = getPoints(P12; permute=false)
    dropdups = Dict{Vector{Int},Int}()
    for sidx = 1:N
        sl1_ = sl1[sidx] % N
        sl1_ = sl1_ == 0 ? N : sl1_
        sl2_ = sl2[sidx] % N
        sl2_ = sl2_ == 0 ? N : sl2_
        u1 = pts1[sl1_]
        u2 = pts2[sl2_]
        # u1 = pts1[sl1[sidx] % N]
        # u2 = pts2[sl2[sidx] % N]

        u12, c12 = calcProductGaussians(M, [u1, u2], [bw1, bw2])
        
        # workaround for duplicate selections in pts12 test
        if !haskey(dropdups, sl[sidx])
          dropdups[sl[sidx]] = length(keys(dropdups)) + 1 #get(dropdups, sl[sidx], 0) + 1
        end
        idxoff = dropdups[sl[sidx]]
        # idxoff = 0
        # for (k,i) in dropdups
        #     idxoff += i
        # end
        # TODO test that kernel weights increase for each duplicate selection
        # @isapprox( getWeights(P12)[invpermute(P12, sidx)], 1 / N * dropdups[sl[sidx]])

        if idxoff <= length(pts12)
            @test isapprox(u12, pts12[idxoff])
        end
    end
end

