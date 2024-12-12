include("fgNT.jl")

function backtracking(t, beta_curr, c_curr, delta_beta, delta_c, paramLS, paramf)
    alpha_LS = paramLS[1];
    gamma_LS = paramLS[2];

    t_NT = 1;
    #f_curr = fval2(t, beta_curr, c_curr, paramf);
    f_curr = fval2(t, beta_curr, c_curr, paramf);
    gradb, gradc = fgrad2(t, beta_curr, c_curr, paramf);

    count = 0;

    # find a step size t_NT such that (beta_new, c_new) is feasible
    while(true)
        count = count + 1;
        if(count>1000)
            println("cann't make the new point feasible")
            break;
        end
        beta_new = beta_curr.+(t_NT.*delta_beta);
        c_new = c_curr.+(t_NT.*delta_c);
        l = (-1).*beta_new.-c_new;
        u = beta_new.-c_new;
        if (if_domain(l, u))
            break;
        else
            t_NT = gamma_LS * t_NT;
        end
    end

    # backtracking LS
    count = 0;
    while(true)
        count = count + 1;
        if(count>1000)
            println("can't find a step size")
            break;
        end
        # new beta & new c
        beta_new = beta_curr.+(t_NT.*delta_beta);
        c_new = c_curr.+(t_NT.*delta_c);

        f_new = fval2(t, beta_new, c_new, paramf);

        # check sufficient reduction condition
        if(f_new <= f_curr + alpha_LS*t_NT*sum((gradb'*delta_beta).+(gradc'*delta_c)))
            break;
        end
        t_NT = gamma_LS* t_NT;
    end

    return t_NT
end

# check whether (beta, c) is feasible
function if_domain(l, u)
    if ((minimum(Int.(l.<0))>0)&(minimum(Int.(u.<0))>0))
        return true
    else
        return false
    end
end
