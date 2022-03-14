import numpy as np
import math

def multiplicative_update_gaussian(X, k, num_iterations):

    # epsilon to avoid division by zero
    epsilon = 1e-6
    m, N = X.shape
    
    # randomly initialize A, S from a Gaussian distribution
    A = np.random.randn(m, k)
    S = np.random.randn(k, N)
    
    loss = []

    # interatively update factorizations
    for i in range(num_iterations):
        A = A * ((X @ S.T) / (A @ S @ S.T + epsilon))
        S = S * ((A.T @ X) / (A.T @ A @ S + epsilon))
        loss.append(np.linalg.norm(X - A @ S)**2)
    
    return A, S, loss

