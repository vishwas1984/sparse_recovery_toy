include("fgNT.jl")
include("backtracking.jl")

function NT_mtd(t, beta_init, c_init, eps_NT, paramLS, paramf)
    beta = beta_init;
    c = c_init;

    count = 0;

    # workspace for Krylov.cg!
    n = length(beta)
    if gpu
        workspace = Krylov.CgSolver(2*n, 2*n, CuVector{Float64})
        rhs = CuVector{Float64}(undef, 2*n)
    else
        workspace = Krylov.CgSolver(2*n, 2*n, Vector{Float64})
        rhs = Vector{Float64}(undef, 2*n)
    end
    op_FFT = plan_fft(workspace.x[1:n])

    while(true)
        count = count + 1;
        if count > 1000
            println("Newton's method doesn't terminate, t = ", t)
        end

        gradb, gradc = fgrad2(t, beta, c, paramf);
        if gpu
            rhs[1:n] .= -1.0 .* CuVector{Float64}(gradb)
            rhs[(n+1):(2*n)] .= -1.0 .* CuVector{Float64}(gradc)
        else
            rhs[1:n] .= -1.0 .* gradb
            rhs[(n+1):(2*n)] .= -1.0 .* gradc
        end

        # delta_beta, delta_c = CG(workspace, t, beta, c, gradb, gradc, paramf);

        beta_tmp = gpu ? CuVector(beta) : beta
        c_tmp = gpu ? CuVector(c) : c
        t_start = time()
        delta_beta, delta_c = CG_alexis(workspace, op_FFT, t, beta_tmp, c_tmp, rhs, paramf);
        t_end = time()
        elapsed_time = t_end - t_start
        println("It requires $(elapsed_time) seconds to solve system with CG at iteration $count.")

        delta_beta = Vector{Float64}(delta_beta)
        delta_c = Vector{Float64}(delta_c)

        #delta_beta1, delta_c1 = NTdir(t, beta, c, gradb, gradc, paramf);
        #println("normdeltab:", norm(delta_beta.-delta_beta1))
        #println("normdeltac:", norm(delta_c.-delta_c1))

        # compute increment
        lambda2 = -sum((gradb'*delta_beta).+(gradc'*delta_c));

        # check terminate condition
        if ((lambda2/2) <= eps_NT)
            break;
        end

        # use bracktracking line search to compute step size
        t_NT = backtracking(t, beta, c, delta_beta, delta_c, paramLS, paramf);

        # update beta, c
        beta = beta.+(t_NT.*delta_beta);
        c = c.+(t_NT.*delta_c);
    end

    return beta, c
end
