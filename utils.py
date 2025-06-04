import numpy as np
from scipy.linalg import hankel

'''
Some useful tools for DOA estimation
'''


def beamforming_2D(Y, array_x, array_y, scan_resolution):
    # Define the azimuth and elevation grids
    grid_a = np.arange(-90, 90 + scan_resolution, scan_resolution)
    grid_e = np.arange(-90, 90 + scan_resolution, scan_resolution)

    beampattern = np.zeros((len(grid_a), len(grid_e)))

    N_1 = len(array_x)
    N_2 = len(array_y)

    for i, theta_a in enumerate(grid_a):
        for k, theta_e in enumerate(grid_e):
            # Compute the manifold vectors
            manifold_i = np.exp(1j * np.pi * array_x[:, np.newaxis] * np.sin(np.radians(theta_e)))
            manifold_k = np.exp(1j * np.pi * array_y[:, np.newaxis] * np.sin(np.radians(theta_a)))

            # Compute the Kronecker product of the manifolds
            manifold = np.kron(manifold_i, manifold_k)

            # Compute the beam pattern value
            beampattern[i, k] = np.abs(manifold.T.conj() @ Y.flatten()) / (N_1 * N_2)

    return beampattern, grid_a, grid_e

def ssfbmusic_2d(K, SS_matrix, S_bx, S_by):
    M, N = SS_matrix.shape

    # Compute the covariance matrix
    covariance_X = (SS_matrix @ SS_matrix.conj().T) / N

    # Compute the forward-backward averaged covariance matrix
    J = np.fliplr(np.eye(M))
    covariance_X_FB = 0.5 * (covariance_X + J @ covariance_X.conj() @ J)

    # Perform eigenvalue decomposition
    D, V = np.linalg.eig(covariance_X_FB)
    ind = np.argsort(D)[::-1]
    V_sorted = V[:, ind]

    # Noise subspace
    W = V_sorted[:, K:]

    # Define the azimuth and elevation grids
    grid_a = np.arange(-90, 91, 1)
    grid_e = np.arange(-90, 91, 1)

    value_record = np.zeros((len(grid_e), len(grid_a)))

    # Compute the pseudo-spectrum
    for i, theta_e in enumerate(grid_e):
        for k, theta_a in enumerate(grid_a):
            # Compute manifold vectors
            manifold_i = np.exp(1j * np.pi * S_bx[:, np.newaxis] * np.sin(np.radians(theta_e)))
            manifold_k = np.exp(1j * np.pi * S_by[:, np.newaxis] * np.sin(np.radians(theta_a)))

            # Compute Kronecker product
            manifold = np.kron(manifold_i, manifold_k)

            # Compute pseudo-spectrum value
            value_ik = 1 / np.abs(manifold.conj().T @ (W @ W.conj().T) @ manifold)
            value_record[i, k] = value_ik

    return value_record, grid_a, grid_e

def block_hankel(X):
    M, N = X.shape
    
    p1 = M // 2 + 1
    p2 = N // 2 + 1

    # Create all Hankel matrices for each row of X
    All_hankelx = np.zeros((M * p2, N - p2 + 1), dtype=X.dtype)
    for k in range(M):
        x_k = X[k, :]
        hankelx_k = hankel(x_k[:p2], x_k[p2-1:])
        All_hankelx[k * p2:(k + 1) * p2, :] = hankelx_k

    # Assemble the block Hankel matrix
    block_hankelX = np.zeros((p1 * p2, (M - p1 + 1) * (N - p2 + 1)), dtype=X.dtype)
    for l in range(p1):
        for m in range(M - p1 + 1):
            ind = l + m
            block_hankelX[l * p2:(l + 1) * p2, m * (N - p2 + 1):(m + 1) * (N - p2 + 1)] = \
                All_hankelx[ind * p2:(ind + 1) * p2, :]

    return block_hankelX, p1, p2
