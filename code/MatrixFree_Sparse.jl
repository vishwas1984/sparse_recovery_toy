using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
using Plots;
using Ipopt, JuMP;
using Plots;
using BenchmarkTools
include("IPmtd.jl")
include("Mperptz.jl")
include("Mtgeneration.jl")
include("punching.jl")
include("Admm_MatrixFree.jl")
using LaplaceInterpolation
using NPZ


function punch_3D_cart(center, radius, x, y, z; linear = false)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    inds = filter(i -> (((x[i[1]]-center[1])/radius_x)^2 
                        + ((y[i[2]]-center[2])/radius_y)^2 
                        + ((z[i[3]] - center[3])/radius_z)^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    (length(inds) == 0) && error("Empty punch.")
    if linear == false
      return inds
    else
      return LinearIndices(zeros(length(x), length(y), length(z)))[inds]
    end 
end

z3d = npzread("z3d_movo.npy");
dx = 0.02
dy = 0.02
dz = 0.02
x = -0.2:dx:4.01
y = -0.2:dy:6.01
z = -0.2:dz:6.01
x = x[1:210]
y = y[1:310]
z = z[1:310]

radius = 0.2001
punched_pmn = copy(z3d)
punched_pmn = punched_pmn[1:210, 1:310, 1:310]
index_missing_2D = CartesianIndex{3}[]
for i=0:4.
    for j=0:6.
        for k = 0:6.
            center =[i,j,k];
            absolute_indices1 = punch_3D_cart(center, radius, x, y, z);
            punched_pmn[absolute_indices1] .= 0;
            append!(index_missing_2D, absolute_indices1);
        end
    end
end

# print(index_missing_2D)



function paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)
    paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
    paramLS = (alpha_LS, gamma_LS)
    paramB = (eps_barrier, mu_barrier)
    paramset = (paramB, eps_NT, paramLS, paramf)
    return paramset
end



DFTsize = size(punched_pmn); # problem dim
DFTdim = length(DFTsize); # problem size
M_perptz = M_perp_tz(DFTdim, DFTsize, punched_pmn);
Nt = prod(DFTsize)

lambda = 1;

alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 1e-6;
eps_barrier = 1e-6;
mu_barrier = 10;

paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_2D, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)


t_init = 1;
beta_init = ones(Nt)./2;
c_init = ones(Nt);

beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)


# beta_true = DFT_to_beta(DFTdim, DFTsize, w);
# sum(abs.(beta_true))

# Nt = 500;
# t = collect(0:(Nt-1));
# x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
# x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
# x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
# x = x1.+x2.+x3; #signal
# Random.seed!(1)
# y = x + randn(Nt); #noisy signal

# w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
# DFTsize = size(x); # problem dim
# DFTdim = length(DFTsize); # problem size

# missing_prob = 0.15
# centers = centering(DFTdim, DFTsize, missing_prob)
# radius = 1
# index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz

# lambda = 100;

# alpha_LS = 0.1;
# gamma_LS = 0.8;
# eps_NT = 1e-6;
# eps_barrier = 1e-6;
# mu_barrier = 10;


# paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

# t_init = 1;
# beta_init = ones(Nt)./2;
# c_init = ones(Nt);

# beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)

# rho = 1
# paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
# beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)

# norm(beta_IPOPT.-beta_ADMM)

# plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), title = "IP: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

# plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), title = "ADMM: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)


# ## 2d
# Nt = 20;
# Ns = 24;
# t = collect(0:(Nt-1));
# s = collect(0:(Ns-1));
# x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
# Random.seed!(1)
# y = x + randn(Nt,Ns)#noisy signal

# w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
# DFTsize = size(x); # problem dim
# DFTdim = length(DFTsize); # problem size
# beta_true = DFT_to_beta(DFTdim, DFTsize, w);
# sum(abs.(beta_true))

# # randomly generate missing indices
# missing_prob = 0.15
# centers = centering(DFTdim, DFTsize, missing_prob)
# radius = 1

# index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


# # unify parameters for barrier method
# M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero);
# lambda = 5

# paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

# t_init = 1;
# beta_init = zeros(prod(DFTsize));
# c_init = ones(prod(DFTsize));

# beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)

# rho = 1
# paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian)
# beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)

# norm(beta_IPOPT.-beta_ADMM)

# plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 2d (20*24) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 2d (20*24) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), title = "IP: 2d (20*24) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

# plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 2d (20*24) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 2d (20*24) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), title = "ADMM: 2d (20*24) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)



# ## 3d
# N1 = 6;
# N2 = 8;
# N3 = 10;
# idx1 = collect(0:(N1-1));
# idx2 = collect(0:(N2-1));
# idx3 = collect(0:(N3-1));
# x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
# Random.seed!(2)
# y = x + rand(N1, N2, N3); # noisy signal

# w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
# DFTsize = size(x); # problem dim
# DFTdim = length(DFTsize); # problem size
# beta_true = DFT_to_beta(DFTdim, DFTsize, w);
# sum(abs.(beta_true))


# # randomly generate missing indices
# missing_prob = 0.15
# centers = centering(DFTdim, DFTsize, missing_prob)
# radius = 1

# index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# # unify parameters for barrier method
# M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero);
# lambda = 5

# paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

# t_init = 1;
# beta_init = zeros(prod(DFTsize));
# c_init = ones(prod(DFTsize));

# beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)

# rho = 1
# paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian)
# beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)

# norm(beta_IPOPT.-beta_ADMM)

# plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_IPOPT), title = "IP: 3d (6*8*10) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

# plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 3d (6*8*10) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 3d (6*8*10) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
# plot(log.(subgrad_ADMM), title = "ADMM: 3d (6*8*10) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
