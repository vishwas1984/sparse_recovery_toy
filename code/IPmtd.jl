include("Newton.jl")
include("subgrad.jl")

function barrier_mtd(beta_init, c_init, t_init, paramset)
    paramB = paramset[1];
    eps_NT = paramset[2];
    paramLS = paramset[3];
    paramf = paramset[4];

    eps_barrier = paramB[1];
    mu_barrier = paramB[2];

    beta = beta_init;
    c = c_init;
    t = t_init;

    n = length(beta_init);

    println("barrier started")
    subgrad_vec = [subgrad(paramf, beta)];
    time_vec = [0];

    Time = time();
    count = 0;
    while(true)
        count = count + 1;
        if(count>1000)
            println("barrier method doesn't terminate");
        end
        # Newton step
        beta, c = NT_mtd(t, beta, c, eps_NT, paramLS, paramf)
        subgrad_vec = [subgrad_vec; subgrad(paramf, beta)];
        time_vec = [time_vec; time()-Time];

        # Check termination condition
        if((((2n)/t) < eps_barrier)||(t>10^7))
            break;
        end

        # Update t
        t = mu_barrier * t;

    end

    # compute the average time cost for the Newton step computation
    #timeave = mean(timevec);

    return beta, c, subgrad_vec, time_vec
end
