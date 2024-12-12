using LinearAlgebra;
using SparseArrays;

include("Mperptz.jl")
# include("Mperptz_rfft.jl")

function fval2(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
#    Mt = paramf[6];

    l = (-1).*beta.-c;
    u = beta.-c;

    fval = (t/2)*sum((M_perp_beta_old(DFTdim, DFTsize, beta, index_missing)).^2) - t*beta'*M_perptz + t*lambda*sum(c)- sum(log.((-1).*l)) - sum(log.((-1).*u));
    return fval
end


function fgrad2(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
#    Mt = paramf[6];

    n = length(beta);

    l = (-1).*beta.-c;
    u = beta.-c;

    gradb = t.*(M_perpt_M_perp_vec_old(DFTdim, DFTsize, beta, index_missing).-Mperptz).+inv.(l).-inv.(u);
    gradc = (t*lambda).*ones(n).+inv.(l).+inv.(u);

    return gradb, gradc
end

function L_inv_r(t, l, u, r)
    n = length(l);
    r_beta = r[1:n];
    r_c = r[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2)).+t;
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2));

    l22_inv = inv.(l22.-((l12.^2).*(inv.(l11))));
    l12_inv = (inv.(l11)).*l12.*l22_inv.*(-1);
    l11_inv = (inv.(l11)).+((inv.(l11)).^2).*(l12.^2).*l22_inv;

    return [(l11_inv.*r_beta).+(l12_inv.*r_c); (l12_inv.*r_beta).+(l22_inv.*r_c)]
end

function Hessian_vec(t, l, u, dim, size, idx_missing, p)
    n = length(l);
    p_beta = p[1:n];
    p_c = p[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2));
    l12 = (inv.(l.^2)).-(inv.(u.^2));

    H11_pbeta = (l11.*p_beta).+((M_perpt_M_perp_vec(dim, size, p_beta, idx_missing)).*t);
    H12_pc = l12.*p_c;
    H21_pbeta = l12.*p_beta;
    H22_pc = l11.*p_c;

    return [H11_pbeta.+H12_pc; H21_pbeta.+H22_pc]
end

function CG(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    dim = paramf[1];
    size = paramf[2];
    lambda = paramf[4];
    idx_missing = paramf[5];
    #Mt = paramf[6];

    l = (-1).*beta.-c;
    u = beta.-c;
    n = length(beta);

    b = [gradb; gradc].*(-1);
    x0 = zeros(2*n);
    r0 = Hessian_vec(t, l, u, dim, size, idx_missing, x0).-b;
    y0 = L_inv_r(t, l, u, r0);
    p0 = y0.*(-1);

    #iter = 0
    while(norm(r0)>CG_esp)
        #iter = iter + 1;
        Hes_p = Hessian_vec(t, l, u, dim, size, idx_missing, p0);
        alpha = (r0'*y0)/(p0'*Hes_p);
        #println("t=", t, " norm =", norm(Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, p0)-Hessian_vec(t, l, u, h, g, Mt, p0)))
        x1 = x0.+(p0.*alpha);
        r1 = r0.+(Hes_p.*alpha);
        y1 = L_inv_r(t, l, u, r1);
        beta = (r1'*y1)/(r0'*y0);
        p0 = (p0.*beta).-y1;
        r0 = r1;
        y0 = y1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)]#, iter
end

function L_inv_r_alexis(y, n, l11_inv, l12_inv, l22_inv, r)
    r_beta = view(r, 1:n)
    r_c = view(r, (n+1):(2*n))

    y_beta = view(y, 1:n)
    y_c = view(y, (n+1):(2*n))

    y_beta .= (l11_inv .* r_beta) .+ (l12_inv .* r_c)
    y_c .= (l12_inv .* r_beta) .+ (l22_inv .* r_c)
    return y
end

function Hessian_vec_alexis(y, op_FFT, n, t, l, u, dim, size, idx_missing, p)
    p_beta = view(p, 1:n)
    p_c = view(p, (n+1):(2*n))

    y_beta = view(y, 1:n)
    y_c = view(y, (n+1):(2*n))

    l11 = (inv.(l.^2)) .+ (inv.(u.^2));
    l12 = (inv.(l.^2)) .- (inv.(u.^2));

    if gpu
        H11_pbeta = (l11 .* p_beta) .+ (M_perpt_M_perp_vec_gpu(op_FFT, dim, size, p_beta, idx_missing) .* t);
    else
        H11_pbeta = (l11 .* p_beta) .+ (M_perpt_M_perp_vec_old(dim, size, p_beta, idx_missing) .* t);
        # H11_pbeta = (l11 .* p_beta) .+ (M_perpt_M_perp_vec(op_FFT, dim, size, p_beta, idx_missing) .* t);
    end
    H12_pc = l12 .* p_c;
    H21_pbeta = l12.* p_beta;
    H22_pc = l11 .* p_c;

    y_beta .= H11_pbeta .+ H12_pc
    y_c .= H21_pbeta .+ H22_pc
    return y
end

function CG_alexis(workspace, op_FFT, t, beta, c, rhs, paramf, CG_esp = 10e-6)
    global nkrylov_ipm += 1

    dim = paramf[1];
    size = paramf[2];
    lambda = paramf[4];
    idx_missing = paramf[5];
    #Mt = paramf[6];

    l = -beta .- c;
    u = beta .- c;
    n = length(beta);

    l11 = (inv.(l.^2)) .+ (inv.(u.^2)) .+ t;
    l12 = (inv.(l.^2)) .- (inv.(u.^2));
    l22 = (inv.(l.^2)) .+ (inv.(u.^2));
    l22_inv = inv.(l22 .- ((l12.^2) .* (inv.(l11))));
    l12_inv = (inv.(l11)) .* l12.*l22_inv .* (-1);
    l11_inv = (inv.(l11)) .+ ((inv.(l11)).^2) .* (l12.^2) .* l22_inv;

    # A = LinearOperator(Float64, 2*n, 2*n, true, true, (y, v) -> (y .= Hessian_vec_alexis(t, l, u, dim, size, idx_missing, v)))
    A = LinearOperator(Float64, 2*n, 2*n, true, true, (y, v) -> Hessian_vec_alexis(y, op_FFT, n, t, l, u, dim, size, idx_missing, v))
    # P = LinearOperator(Float64, 2*n, 2*n, true, true, (y, v) -> (y .= L_inv_r(t, l, u, v)))
    P = LinearOperator(Float64, 2*n, 2*n, true, true, (y, v) -> L_inv_r_alexis(y, n, l11_inv, l12_inv, l22_inv, v))
    Krylov.cg!(workspace, A, rhs, M=P, atol=CG_esp, rtol=0.0, verbose=0)

    x = Krylov.solution(workspace)
    return x[1:n], x[(n+1):(2*n)]#, iter
end
