using LinearAlgebra;
using Random, Distributions;
# generate matrix M

# @param dim The dimension of the problem (dim = 1, 2, 3)
# @param size The size of each dimension of the problem
#(we only consider the cases when the sizes are even for all the dimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))
# @param missing_idx If dim = 1, missing_idx is an array consist of all the missing indices, e.g. missing_idx = [2;3;5];
# If dim = 2 or 3, missing_idx is an array consist of the Cartesian indices of all the missing indices
# e.g. missing_idx = [CartesianIndex(4,2); CartesianIndex(2,3); CartesianIndex(1,5)];

# @details This fucnction generates the transpose of matrix M

# @return The transpose of matrix M; M corresponds to the missing indices in the signals
# @example
# >dim = 2;
# >size2 = (6, 8);
# >missing_idx = [CartesianIndex(4,2); CartesianIndex(2,3); CartesianIndex(1,5)];
# >Mt = generate_Mt(dim, size2, missing_idx)

function generate_Mt(dim, size, missing_idx)
    if (dim == 1)
        return generate_Mt_1D(missing_idx, size)
    elseif (dim == 2)
        return generate_Mt_2D(missing_idx, size)
    else
        return generate_Mt_3D(missing_idx, size)
    end
end

# 1 dim
function generate_Mt_1D(missing_idx, size)
    N = size[1];
    m = length(missing_idx);
    Mt = Array{Float64, 2}(undef, N, m);

    colcount = 0;
    # generate M^t column by column
    for j in missing_idx
        colcount = colcount + 1;
        Mt[:,colcount] = generate_Mt_row_1D(j, N);
    end
    return Mt
end

# generate one column of M^t
function generate_Mt_row_1D(j, N)
    omega_j = exp(2*pi*im*(j-1)/N);
    Mt_row = (1/sqrt(N)).*[1.0;
                           (-1.0)^(j-1);
                           sqrt(2).*real.(omega_j .^((2-1):(Int(N/2-1))));
                           (-sqrt(2)).*imag.(omega_j .^((2-1):(Int(N/2-1))))];
    return Mt_row
end

# 2 dim
function generate_Mt_2D(missing_idx, size)
    N1 = size[1];
    N2 = size[2];
    m = length(missing_idx);
    Mt = Array{Float64, 2}(undef, N1*N2, m);

    colcount = 0;
    # generate M^t column by column
    for idx in Tuple.(missing_idx)
        colcount = colcount + 1;
        Mt[:,colcount] = generate_Mt_row_2D(idx, N1, N2);
    end
    return Mt
end

# generate one column of M^t
function generate_Mt_row_2D(idx, N1, N2)
    (j,k) = idx;
    omega1_j = exp(2*pi*im*(j-1)/N1);
    omega2_k = exp(2*pi*im*(k-1)/N2);

    Mt_row = Array{Float64, 1}(undef, N1*N2);
    Mt_row[1:4] = [1; (-1)^(k-1); (-1)^(j-1); (-1)^(j-1+k-1)];
    Mt_row[5:Int(4+2*(N2/2-1))] = [sqrt(2).*(real.(omega2_k .^((2-1):(Int(N2/2)-1))));
                                   (-sqrt(2)).*(imag.(omega2_k .^((2-1):(Int(N2/2)-1))))];
    Mt_row[Int(4+2*(N2/2-1)+1):Int(4+4*(N2/2-1))] = ((-1)^(j-1)).*(Mt_row[5:Int(4+2*(N2/2-1))]);
    Mt_row[Int(4+4*(N2/2-1)+1):Int(4+4*(N2/2-1)+2*(N1/2-1))] = [sqrt(2).*(real.(omega1_j .^((2-1):(Int(N1/2)-1))));
                                                                (-sqrt(2)).*(imag.(omega1_j .^((2-1):(Int(N1/2)-1))))];
    Mt_row[Int(4+4*(N2/2-1)+2*(N1/2-1)+1):Int(4+4*(N2/2-1)+4*(N1/2-1))] = ((-1)^(k-1)).*(Mt_row[Int(4+4*(N2/2-1)+1):Int(4+4*(N2/2-1)+2*(N1/2-1))]);
    Mt_row[Int(4+4*(N2/2-1)+4*(N1/2-1)+1):Int(N1*N2)] = [sqrt(2).*(reshape(real.((omega1_j .^((2-1):Int(N1/2-1)))*(transpose(omega2_k .^((2-1):Int(N2/2-1))))), Int((N2/2-1)*(N1/2-1)), 1));
                                                         (-sqrt(2)).*(reshape(imag.((omega1_j .^((2-1):Int(N1/2-1)))*(transpose(omega2_k .^((2-1):Int(N2/2-1))))), Int((N2/2-1)*(N1/2-1)), 1));
                                                         sqrt(2).*(reshape(real.((omega1_j .^((2-1):Int(N1/2-1)))*(transpose(omega2_k .^(Int(N2/2+2-1):Int(N2-1))))), Int((N2/2-1)*(N1/2-1)), 1));
                                                         (-sqrt(2)).*(reshape(imag.((omega1_j .^((2-1):Int(N1/2-1)))*(transpose(omega2_k .^(Int(N2/2+2-1):Int(N2-1))))), Int((N2/2-1)*(N1/2-1)), 1))];
    Mt_row = (1/sqrt(N1*N2)).*Mt_row;
    return Mt_row
end

# 3 dim
function generate_Mt_3D(missing_idx, size)
    N1 = size[1];
    N2 = size[2];
    N3 = size[3];
    m = length(missing_idx);
    Mt = Array{Float64, 2}(undef, N1*N2*N3, m);

    colcount = 0;
    # generate M^t column by column
    for idx in Tuple.(missing_idx)
        colcount = colcount + 1;
        Mt[:,colcount] = generate_Mt_row_3D(idx, N1, N2, N3);
    end
    return Mt
end

# generate one column of M^t
function generate_Mt_row_3D(idx, N1, N2, N3)
    (j,k,l) = idx;
    omega1_j = exp(2*pi*im*(j-1)/N1);
    omega2_k = exp(2*pi*im*(k-1)/N2);
    omega3_l = exp(2*pi*im*(l-1)/N3);

    Mt_row = Array{Float64, 1}(undef, N1*N2*N3);
    Mt_row[1:8] = [1; (-1)^(l-1); (-1)^(k-1); (-1)^(k-1+l-1); (-1)^(j-1); (-1)^(j-1+l-1); (-1)^(j-1+k-1); (-1)^(j-1+k-1+l-1)];

    Mt_row[Int(8+1):Int(8+2*(N3/2-1))] = [sqrt(2).*(real.(omega3_l .^((2-1):(Int(N3/2-1)))));
                                          (-sqrt(2)).*(imag.(omega3_l .^((2-1):(Int(N3/2-1)))))];

    Mt_row[Int(8+2*(N3/2-1)+1):Int(8+8*(N3/2-1))] = [((-1)^(k-1)).*(Mt_row[Int(8+1):Int(8+2*(N3/2-1))]);
                                                     ((-1)^(j-1)).*(Mt_row[Int(8+1):Int(8+2*(N3/2-1))]);
                                                     ((-1)^(j-1+k-1)).*(Mt_row[Int(8+1):Int(8+2*(N3/2-1))])];

    Mt_row[Int(8+8*(N3/2-1)+1):Int(8+8*(N3/2-1)+2*(N2/2-1))] = [sqrt(2).*(real.(omega2_k .^((2-1):(Int(N2/2-1)))));
                                                                (-sqrt(2)).*(imag.(omega2_k .^((2-1):(Int(N2/2-1)))))];
    Mt_row[Int(8+8*(N3/2-1)+2*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1))] = [((-1)^(l-1)).*(Mt_row[Int(8+8*(N3/2-1)+1):Int(8+8*(N3/2-1)+2*(N2/2-1))]);
                                                                           ((-1)^(j-1)).*(Mt_row[Int(8+8*(N3/2-1)+1):Int(8+8*(N3/2-1)+2*(N2/2-1))]);
                                                                           ((-1)^(j-1+l-1)).*(Mt_row[Int(8+8*(N3/2-1)+1):Int(8+8*(N3/2-1)+2*(N2/2-1))])];

    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+2*(N1/2-1))] = [sqrt(2).*(real.(omega1_j .^((2-1):(Int(N1/2-1)))));
                                                                                      (-sqrt(2)).*(imag.(omega1_j .^((2-1):(Int(N1/2-1)))))];
    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+2*(N1/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1))] = [((-1)^(l-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+2*(N1/2-1))]);
                                                                                                 ((-1)^(k-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+2*(N1/2-1))]);
                                                                                                 ((-1)^(k-1+l-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+2*(N1/2-1))])];

    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+4*(N2/2-1)*(N3/2-1))] = [sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)))));
                                                                                                                    (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)))));
                                                                                                                    sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)))));
                                                                                                                    (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)))))];
    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+4(N2/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1))] = ((-1)^(j-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+4*(N2/2-1)*(N3/2-1))]);

    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+4*(N1/2-1)*(N3/2-1))] = [sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                                           (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                                           sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))));
                                                                                                                                                           (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))))];
    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+4*(N1/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1))] = ((-1)^(k-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+4*(N1/2-1)*(N3/2-1))]);

    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+4*(N1/2-1)*(N2/2-1))] = [sqrt(2).*(real.(kron(omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                                                                                   (-sqrt(2)).*(imag.(kron(omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                                                                                   sqrt(2).*(real.(kron(omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))));
                                                                                                                                                                                                   (-sqrt(2)).*(imag.(kron(omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))))];
    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+4*(N1/2-1)*(N2/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+8*(N1/2-1)*(N2/2-1))] = ((-1)^(l-1)).*(Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+1):Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+4*(N1/2-1)*(N2/2-1))]);

    Mt_row[Int(8+8*(N3/2-1)+8*(N2/2-1)+8*(N1/2-1)+8(N2/2-1)*(N3/2-1)+8*(N1/2-1)*(N3/2-1)+8*(N1/2-1)*(N2/2-1)+1):Int(N1*N2*N3)] = [sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                  (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                  sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))));
                                                                                                                                  (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^((2-1):Int(N2/2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))));
                                                                                                                                  sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                  (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)), omega1_j .^((2-1):Int(N1/2-1)))));
                                                                                                                                  sqrt(2).*(real.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))));
                                                                                                                                  (-sqrt(2)).*(imag.(kron(omega3_l .^((2-1):Int(N3/2-1)), omega2_k .^(Int(N2/2+2-1):Int(N2-1)), omega1_j .^(Int(N1/2+2-1):Int(N1-1)))))];

    Mt_row = (1/sqrt(N1*N2*N3)).*Mt_row;
    return Mt_row
end
