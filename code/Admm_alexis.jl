include("subgrad.jl")
include("Mperptz.jl")

import Krylov
using LinearOperators

function cgADMM_alexis(paramf, rho, maxt = 10000, tol = 1e-6)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    missing_indices = paramf[5];
    #Mt = paramf[6];

    n = prod(DFTsize);
    x0 = ones(n);
    z0 = ones(n);
    y0 = zeros(n);
    t = 0;
    err = 1;
    subgrad_vec = subgrad(paramf, z0);
    time_vec = [0];

    # Model M_perp^t M_perp + rho I
    op = LinearOperator(Float64, n, n, true, true, (y, v) -> (y .= M_perptM_perp_rho_x(missing_indices, rho, v, DFTdim, DFTsize)))
    solver = Krylov.CgSolver(n, n, Vector{Float64})

    Time = time();
    while (t < maxt) && (err > tol)
        b = M_perptz .+ (rho .* z0) .- y0;

        # update x
        Krylov.cg!(solver, op, b, itmax=maxt, atol=tol, rtol=0.0);
        x1 = solver.x

        # update z
        z1 = softthreshold.(x1 + y0/rho, lambda/rho);

        # update y
        y1 = y0 + rho * (x1 - z1);

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

function M_perptM_perp_rho_x(missing_indices, rho, x, DFTdim, DFTsize)
    # return (M_perp^t * M_perp + rho * I) * x
    n = prod(DFTsize);
    Ax = rho .* x;
    w = M_perpt_M_perp_vec_old(DFTdim, DFTsize, x, missing_indices)
    Ax = w .+ Ax;
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
