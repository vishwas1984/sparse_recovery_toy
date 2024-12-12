using LinearAlgebra;
using FFTW;
using Random, Distributions;

include("mapping.jl")
include("mapping_gpu.jl")

# compute M_{\perp}^{\top}z

# @param z_zero The zero-imputed signal, i.e. replacing all the missing values in the signal with 0.
# e.g. The signal is [2;3;missing;4], then z_zero = [2;3;0;4].
# @param dim The dimension of the problem (dim = 1, 2, 3)
# @param size The size of each dimension of the problem
#(we only consider the cases when the sizes are even for all the dimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))

# @details This function computes M_{\perp}^{\top}z.

# @return M_{\perp}^{\top}z A vector with length equal to the product of size
# @example
# >widetildez = [2;3;missing;4];
# >z_zero  = [2;3;0;4];
# >dim = 1;
# >size1 = 4;
# >M_perptz = M_perp_tz(z_zero, dim, size1);

function M_perp_tz_old(dim, size, z_zero)
    N = prod(size);
    temp = fft(z_zero) ./ sqrt(N);
    beta = DFT_to_beta(dim, size, temp)
    return beta
end

function M_perp_tz_old2(dim, size, z_zero)
    N = prod(size);
    beta = real.(rfft(z_zero))
    return beta
end

function M_perp_beta_old(dim, size, beta, idx_missing)
    N = prod(size);
    v = beta_to_DFT(dim, size, beta);
    temp = real.(ifft(v)) .* sqrt(N);
    temp[idx_missing] .= 0;
    return temp
end

function M_perp_beta_old2(dim, size, beta, idx_missing)
    N = prod(size);
    M = 2 * (N - 1)
    temp = irfft(beta, M)
    temp[idx_missing] .= 0;
    return temp
end

function M_perpt_M_perp_vec_old(dim, size, vec, idx_missing)
    temp = M_perp_beta_old(dim, size, vec, idx_missing);
    temp = M_perp_tz_old(dim, size, temp);
    return temp
end

function M_perpt_M_perp_vec_old2(dim, size, vec, idx_missing)
    temp = M_perp_beta_old2(dim, size, vec, idx_missing);
    temp = M_perp_tz_old2(dim, size, temp);
    return temp
end

function M_perp_tz(op_FFT, dim, size, z_zero)
    N = prod(size);
    temp = (op_FFT * z_zero) ./ sqrt(N);
    return DFT_to_beta(dim, size, temp)
end

function M_perp_beta(op_FFT, dim, size, beta, idx_missing)
    N = prod(size);
    v = beta_to_DFT(dim, size, beta);
    temp = real.(op_FFT \ v) .* sqrt(N);
    temp[idx_missing] .= 0;
    return temp
end

function M_perpt_M_perp_vec(op_FFT, dim, size, vec, idx_missing)
    temp = M_perp_beta(op_FFT, dim, size, vec, idx_missing);
    temp = M_perp_tz(op_FFT, dim, size, temp);
    return temp
end

# Note: we should use fft! and ifft!
function M_perp_tz_gpu(op_FFT, dim, size, z_zero)
    N = prod(size);
    temp = (op_FFT * z_zero) ./ sqrt(N);
    return DFT_to_beta_gpu(dim, size, temp)
end

function M_perp_beta_gpu(op_FFT, dim, size, beta, idx_missing)
    N = prod(size);
    v = beta_to_DFT_gpu(dim, size, beta);
    temp = real.(op_FFT \ v) .* sqrt(N);
    temp[idx_missing] .= 0;
    return temp
end

function M_perpt_M_perp_vec_gpu(op_FFT, dim, size, vec, idx_missing)
    temp = M_perp_beta_gpu(op_FFT, dim, size, vec, idx_missing);
    temp = M_perp_tz_gpu(op_FFT, dim, size, temp);
    return temp
end
