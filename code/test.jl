using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;

include("IPmtd.jl")
include("Mperptz.jl")
include("Mtgeneration.jl")
include("punching.jl")
include("Admm_MatrixFree.jl")


function paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)
    paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
    paramLS = (alpha_LS, gamma_LS)
    paramB = (eps_barrier, mu_barrier)
    paramset = (paramB, eps_NT, paramLS, paramf)
    return paramset
end

Nt = 50;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
y = x + randn(Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size

missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz

lambda = 5;

alpha_LS = 0.05;
gamma_LS = 0.8;
eps_NT = 1e-6;
eps_barrier = 1e-6;
mu_barrier = 10;


paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = zeros(Nt);
c_init = ones(Nt);

beta = barrier_mtd(beta_init, c_init, t_init, paramset)[1]

rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
beta_ADMM = cgADMM(paramf, rho)[1]

norm(beta.-beta_ADMM)
