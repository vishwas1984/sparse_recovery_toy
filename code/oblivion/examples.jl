using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
using Ipopt, JuMP;
using Plots;
using NPZ;

include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")
include("backtracking.jl")
include("Newton.jl")
include("IPmtd.jl")
#include("ADMM.jl")
include("Admm_MatrixFree2.jl")

# interior point method parameters
eps_barrier = 10e-6;
mu_barrier = 10;
alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 10e-6;

# unify all the parameters
function param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS)
    paramf = (Mt, M_perptz, d);
    paramLS = (alpha_LS, gamma_LS, paramf);
    paramNT = (eps_NT, paramLS);
    paramB = (eps_barrier, mu_barrier, paramNT);
    return paramB;
end

# 1 dim Interior point and ADMM


# 2 dim Interior point and ADMM

# data generation
Nt = 6;
Ns = 8;


# 3 dim Interior point and ADMM

# data generation
N1 = 6;
N2 = 6;
N3 = 6;
idx1 = collect(0:(N1-1));
idx2 = collect(0:(N2-1));
idx3 = collect(0:(N3-1));
x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
Random.seed!(1)
y = x + rand(N1, N2, N3); # noisy signal

w = round.(fft(x)./sqrt(N1*N2*N3), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size

# randomly generate missing indices
missprob = 0.05; # missing proportion
m = Int(floor(Nt*(missprob))); # number of missing values
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, Int(N1*N2*N3 - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
# note it has to be Cartesian index in 2d and 3d
z_zero = y;
z_zero[index_missing_Cartesian].= 0; # zero-imputed signal

# unify parameters for barrier method
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 225;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, timeave3d = barrier_mtd(beta_init, c_init, t_init, paramB);
println("3d, N1 = 6, N2 = 6, N3 = 6, ave time = ", timeave3d);
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);

#CG-ADMM
lambda = 1;
rho = 1;
#beta_ADMM = cgADMM(Mt, M_perptz, lambda, rho);
beta_ADMM, iter, subgradvec = cgADMM(M_perptz, index_missing_Cartesian, DFTdim, DFTsize, lambda, rho, 10000, 1e-6);
