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

#CG-ADMM
rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_2D)
beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)
