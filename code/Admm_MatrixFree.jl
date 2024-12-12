include("subgrad.jl")
include("Mperptz.jl")

function cgADMM(paramf, rho, maxt = 10000, tol = 1e-6)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    #Mt = paramf[6];


    n = prod(DFTsize);
    x0 = ones(n);
    z0 = ones(n);
    y0 = zeros(n);
    t = 0;
    err = 1;
    subgrad_vec = subgrad(paramf, z0);
    time_vec =[0];

    Time = time();
    while((t<maxt) & (err>tol))
        b = M_perptz.+(rho.*z0).-y0;
        # update x
        x1 = cg(rho, b, index_missing, DFTdim, DFTsize);
        # update z
        z1 = softthreshold.(x1 + y0/rho, lambda/rho);
        # update y
        y1 = y0 + rho*(x1 - z1);
        # check the convergence
        err = max(norm(x1 - x0, 2), norm(y1 - y0, 2), norm(z1 - z0, 2));
        x0 = x1;
        z0 = z1;
        y0 = y1;
        t = t + 1;

        subgrad_vec = [subgrad_vec; subgrad(paramf, z0)];
        time_vec = [time_vec; time()-Time];
    end

    return z0, subgrad_vec, time_vec
end


function cg(rho, b, missing_indices, DFTdim, DFTsize, maxk = 1000, tol = 1e-6)
    n = prod(DFTsize);
    x0 = zeros(n);
    r0 = M_perptM_perp_rho_x(missing_indices, rho, x0, DFTdim, DFTsize).-b;
    p0 = (-1).*r0;
    Ap0 = M_perptM_perp_rho_x(missing_indices, rho, p0, DFTdim, DFTsize);
    k = 0;
    epn = 1;

    while((k<maxk) & (epn>tol))
        alpha = dot(r0, r0)/dot(p0, Ap0);
        x1 = x0 + alpha*p0;
        r1 = r0 + alpha*Ap0;
        beta = dot(r1, r1)/dot(r0, r0);
        p1 = -r1 + beta*p0;

        x0 = x1;
        r0 = r1;
        p0 = p1;
        #Ap0 = ((1+rho).*p0).-(Mt*(Mt'*p0));
        Ap0 = M_perptM_perp_rho_x(missing_indices, rho, p0, DFTdim, DFTsize);

        epn = norm(r0, 2);
        k = k + 1;
    end
    return x0
end

function M_perptM_perp_rho_x(missing_indices, rho, x, DFTdim, DFTsize)
    # return (M_perp^t*M_perp+rho*I)*x
    n = prod(DFTsize);
    Ax = rho.*x;
    w = M_perpt_M_perp_vec_old(DFTdim, DFTsize, x, missing_indices)
    Ax = w.+Ax;
    return Ax
end


function softthreshold(x, thre)
    if(x > thre)
        y = x - thre
    elseif(x < -thre)
        y = x + thre
    else
        y = 0
    end

    return(y)
end
