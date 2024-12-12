# solve \min_||z - M_{\perp}^t*beta||_2 + lambda*||beta||_1

# @param Mt M^{\top}: The transpose of M
# @param M_perptz M_{\perp}^{\top}*z
# @param lambda The penalty parameter in LASSO (scalar)
# @param rho The parameter in ADMM (scalar)
# @param maxk The max number of iters in CG (scalar)
# @param tol The tolerance (scalar)


# @return z0 LASSO problem solution
# @return ADMMerrrecord The norm of the difference between the previous and current step
# It can be deleted. It is saved just for the use of checking convergence of ADMM


function cgADMM(M_perptz, missing_indices, DFTdim, DFTsize, lambda, rho, maxt = 10000, tol = 1e-3)
    n = prod(DFTsize);
    x0 = zeros(n);
    z0 = zeros(n);
    y0 = zeros(n);
    t = 0;
    err = 1;
    subgrad_vec = subgrad(missing_indices, z0, M_perptz, DFTdim, DFTsize, lambda);
    println(last(subgrad_vec));

    while((t<maxt) & (err>tol))
        b = M_perptz.+(rho.*z0).-y0;
        # update x
        x1 = cg(rho, b, missing_indices, DFTdim, DFTsize);
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

        subgrad_vec = [subgrad_vec; subgrad(missing_indices, z0, M_perptz, DFTdim, DFTsize, lambda)];
	println(last(subgrad_vec));
    end

    return z0, t, subgrad_vec
end

# compute (M_{\perp}^{T} M_{\perp}+\rho I)^{-1}*b

# @param Mt M^{\top}: The transpose of M
# @param rho The parameter in ADMM (scalar)
# @param b The vector b = M_{\perp}^t*z.+(rho*z_k).-y_k
# @param maxk The max number of iters in CG (scalar)
# @param tol The tolerance (scalar)


# @return x_0 An approximation to (M_{\perp}^{T} M_{\perp}+\rho I)^{-1}*b
# @return k The number of iterations in CG (can be deleted, saved just for the use of investigating CG and ADMM performance)

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
    w = beta_to_DFT(DFTdim, DFTsize, x);
    w = real.(ifft(w).*sqrt(n));
    w[missing_indices].= 0;
    w = M_perp_tz(w, DFTdim, DFTsize);
    Ax = w.+Ax;
    return Ax
end

function subgrad(missing_indices, z0, M_perptz, DFTdim, DFTsize, lambda)
    n = prod(DFTsize);
    w = beta_to_DFT(DFTdim, DFTsize, z0);
    w = real.(ifft(w).*sqrt(n));
    w[missing_indices].= 0;
    w = M_perp_tz(w, DFTdim, DFTsize);
    w = w.- M_perptz;
    dist_vec = entrywisesubgrad.(w, z0, lambda);
    max_dist = maximum(dist_vec);
    return max_dist
end

function entrywisesubgrad(wi, z0i, lambda)
# compute the distance entrywisely
    if(z0i>1e-6)
        dist = abs(wi+lambda*z0i); #sign of z0 is positive
    elseif(z0i<-1e-6)
        dist = abs(wi-lambda*z0i); #sign of z0 is negative
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

# add soft threshold

# @param x The input value (scalar)
# @param thre The threshold (scalar, thre >= 0)

# @details: if x > thre, then y = x - thre
#           if -thre <= x <= thre, then y = 0
#           if x<= -thre, then y = x + thre

# @return y The value after threshold (scalar)

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
