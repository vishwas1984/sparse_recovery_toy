using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
using Plots;
using Krylov;
using LinearOperators;

include("IPmtd.jl")
include("Mperptz.jl")
include("Mtgeneration.jl")
include("punching.jl")
include("Admm_MatrixFree.jl")

include("Admm_alexis.jl")
version_alexis = false
graphics = false
problem_1d = true
problem_2d = true
problem_3d = true
global gpu = false
global nkrylov_ipm = 0

function paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)
    paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
    paramLS = (alpha_LS, gamma_LS)
    paramB = (eps_barrier, mu_barrier)
    paramset = (paramB, eps_NT, paramLS, paramf)
    return paramset
end

if problem_1d
Nt = 500;
# Nt = 5000;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
# y = x + randn(Nt)  # noisy signal
y = x  # original signal

# w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
w = fft(x) ./ sqrt(Nt)

DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size

# missing_prob = 0.15
missing_prob = 0.0
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

M_perptz = M_perp_tz_old(DFTdim, DFTsize, z_zero); # M_perptz

lambda = 1e-8;
# lambda = 10;

alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 1e-6;
eps_barrier = 1e-6;
mu_barrier = 10;


paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = ones(Nt)./2;
c_init = ones(Nt);

beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)
println("Number of calls to CG: $(nkrylov_ipm).")

rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)

if version_alexis
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM_alexis(paramf, rho)
    t = @elapsed cgADMM_alexis(paramf, rho)
    println("1D -- The first example requires $t seconds.")
else
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)
    t = @elapsed cgADMM(paramf, rho)
    println("1D -- The first example requires $t seconds.")
end

Δβ = norm(beta_IPOPT - beta_ADMM)
println("1D -- ‖β_IPOPT - β_ADMM‖: ", Δβ)

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_ADMM)
Δw = norm(w - w_est)
println("1D -- ‖w - w_est‖: ", Δw)
#############

plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
plot(log.(subgrad_IPOPT), title = "IP: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
plot(log.(subgrad_ADMM), title = "ADMM: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end

if problem_2d
## 2d
# Nt = 20;
# Ns = 24;
Nt = 4;
Ns = 4;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
# y = x + randn(Nt,Ns)  # noisy signal
y = x  # original signal

# w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
w = fft(x) ./ sqrt(Nt*Ns)

DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))

# randomly generate missing indices
# missing_prob = 0.15
missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1

index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


# unify parameters for barrier method
M_perptz = M_perp_tz_old(DFTdim, DFTsize, z_zero);
# lambda = 5
lambda = 1e-8

paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = zeros(prod(DFTsize));
c_init = ones(prod(DFTsize));

beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)

rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian)

if version_alexis
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM_alexis(paramf, rho)
    t = @elapsed cgADMM_alexis(paramf, rho)
    println("2D -- The second example requires $t seconds.")
else
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)
    t = @elapsed cgADMM(paramf, rho)
    println("2D -- The second example requires $t seconds.")
end

Δβ = norm(beta_IPOPT - beta_ADMM)
println("2D -- ‖β_IPOPT - β_ADMM‖: ", Δβ)

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_ADMM)
Δw = norm(w - w_est)
println("2D -- ‖w - w_est‖: ", Δw)
#############

if graphics
plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 2d (20*24) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 2d (20*24) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
plot(log.(subgrad_IPOPT), title = "IP: 2d (20*24) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 2d (20*24) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 2d (20*24) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
plot(log.(subgrad_ADMM), title = "ADMM: 2d (20*24) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end
end

if problem_3d
## 3d
# N1 = 6;
# N2 = 8;
# N3 = 10;

N1 = 4;
N2 = 4;
N3 = 4;

idx1 = collect(0:(N1-1));
idx2 = collect(0:(N2-1));
idx3 = collect(0:(N3-1));
x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
Random.seed!(2)
# y = x + randn(N1, N2, N3)  # noisy signal
y = x  # original signal

# w = round.(fft(x)./sqrt(N1*N2*N3), digits = 4);#true DFT
w = fft(x) ./ sqrt(N1 * N2 * N3)

DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))

# randomly generate missing indices
# missing_prob = 0.15
missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1

index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameters for barrier method
M_perptz = M_perp_tz_old(DFTdim, DFTsize, z_zero);
# lambda = 5
lambda = 1e-8

paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = zeros(prod(DFTsize));
c_init = ones(prod(DFTsize));

beta_IPOPT, c_IPOPT, subgrad_IPOPT, time_IPOPT = barrier_mtd(beta_init, c_init, t_init, paramset)

rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian)

if version_alexis
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM_alexis(paramf, rho)
    t = @elapsed cgADMM_alexis(paramf, rho)
    println("3D -- The third example requires $t seconds.")
else
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)
    t = @elapsed cgADMM(paramf, rho)
    println("3D -- The third example requires $t seconds.")
end

Δβ = norm(beta_IPOPT - beta_ADMM)
println("3D -- ‖β_IPOPT - β_ADMM‖: ", Δβ)

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_ADMM)
Δw = norm(w - w_est)
println("3D -- ‖w - w_est‖: ", Δw)
#############

if graphics
    plot(subgrad_IPOPT, time_IPOPT, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
    plot(log.(subgrad_IPOPT), time_IPOPT, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
    plot(log.(subgrad_IPOPT), title = "IP: 3d (6*8*10) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)

    plot(subgrad_ADMM, time_ADMM, seriestype=:scatter, title = "ADMM: 3d (6*8*10) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
    plot(log.(subgrad_ADMM), time_ADMM, seriestype=:scatter, title = "ADMM: 3d (6*8*10) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
    plot(log.(subgrad_ADMM), title = "ADMM: 3d (6*8*10) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end
end