include("Mperptz.jl")

function subgrad(paramf, beta)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    idx_missing = paramf[5];

    n = prod(DFTsize);
    w = M_perpt_M_perp_vec_old(DFTdim, DFTsize, beta, idx_missing)
    w .= w .- M_perptz;
    dist_vec = entrywisesubgrad.(w, beta, lambda);
    max_dist = maximum(dist_vec);
    return max_dist
end

function entrywisesubgrad(wi, z0i, lambda)
# compute the distance entrywisely
    if(z0i>1e-6)
        #dist = abs(wi+lambda*z0i); #sign of z0 is positive
        dist = abs(wi+lambda);
    elseif(z0i<-1e-6)
        #dist = abs(wi-lambda*z0i); #sign of z0 is negative
        dist = abs(wi-lambda);
    else
        dist = entrywisesubgrad_betaiszero(wi, lambda); #z0 = 0
    end
    return dist
end

function entrywisesubgrad_betaiszero(wi, lambda)
    if(wi<-lambda)
        dist = -lambda.- wi;
    elseif(wi>lambda)
        dist = wi.-lambda;
    else
        dist = 0; #dist = 0
    end
    return dist
end
