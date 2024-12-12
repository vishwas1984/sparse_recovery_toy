using CUDA

function DFT_to_beta_gpu(dim, size, v)
    return DFT_to_beta_1d_gpu(v, size)
end

function kernel_DFT_to_beta_1d!(beta, v, N)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i == 1
        beta[1] = real(v[1])
    elseif i == 2
        beta[2] = real(v[Int(N / 2) + 1])
    elseif 3 <= i <= Int(N / 2) + 1
        idx = i - 2
        beta[i] = sqrt(2) * real(v[idx + 1])
    elseif Int(N / 2) + 2 <= i <= N
        idx = i - Int(N / 2) - 1
        beta[i] = sqrt(2) * imag(v[idx + 1])
    end
    return nothing
end

function DFT_to_beta_1d_gpu(v::CuArray, size)
    N = size[1]
    beta = CUDA.zeros(Float64, N)

    threads_per_block = 512
    blocks = cld(N, threads_per_block)

    @cuda threads=threads_per_block blocks=blocks kernel_DFT_to_beta_1d!(beta, v, N)

    return beta
end

function beta_to_DFT_gpu(dim, size, beta)
    return beta_to_DFT_1d_gpu(beta, size)
end

function kernel_beta_to_DFT_1d!(v, beta, N)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i == 1
        v[1] = beta[1]
    elseif i == Int(N / 2) + 1
        v[Int(N / 2) + 1] = beta[2]
    elseif 3 <= i <= Int(N / 2 + 1)
        idx = i - 2
        v[i] = (beta[i] + im * beta[Int(N / 2 + 1) + idx]) / sqrt(2)
        v[N - idx + 1] = (beta[i] - im * beta[Int(N / 2 + 1) + idx]) / sqrt(2)
    end
    return nothing
end

function beta_to_DFT_1d_gpu(beta, size)
    N = size[1]
    v = CUDA.zeros(ComplexF64, N)

    threads_per_block = 512
    blocks = cld(N, threads_per_block)

    @cuda threads=threads_per_block blocks=blocks kernel_beta_to_DFT_1d!(v, beta, N)

    return v
end
