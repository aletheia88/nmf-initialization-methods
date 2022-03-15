import numpy as np
import math
from scipy.sparse.linalg import svds
from scipy.io import loadmat
from sklearn.cluster import KMeans

def multiplicative_update(X, k, num_iterations, method):

    # epsilon to avoid division by zero
    epsilon = 1e-6
    
    A, S = initialize(X, k, method)

    # interatively update factorizations
    loss = []
    for i in range(num_iterations):
        A = A * ((X @ S.T) / (A @ S @ S.T + epsilon))
        S = S * ((A.T @ X) / (A.T @ A @ S + epsilon))
        loss.append(np.linalg.norm(X - A @ S)**2)
    
    return A, S, loss

def initialize(X, k, method='gaussian'):

    m, N = X.shape
    if method == 'gaussian':
        A = np.random.randn(m, k)
        S = np.random.randn(k, N)
    
    elif method == 'uniform':
        A = np.random.uniform(0, 1, m * k).reshape(m, k)
        S = np.random.uniform(0, 1, k * N).reshape(k, N)

    elif method == 'laplacian':
        A = np.random.laplace(0, 1, m * k).reshape(m, k)
        S = np.random.laplace(0, 1, k * N).reshape(k, N)

    elif method == 'poisson':
        A = np.random.poison(1.0, m * k).reshape(m, k)
        S = np.random.poison(1.0, k * N).reshape(k, N)

    elif method == 'kmeans':
        A, S = kmeans(X, k)

    elif method == 'nndsvd':
        A, S = NNDSVD(X, k)

    return A, S

def kmeans(X, k, version='rand_H'):
    
    # centroids shape = (10, 220)
    centroids = KMeans(n_clusters=k).fit(X.T).cluster_centers_
    m, N = X.shape # (220, 256)
    W = np.empty([m, k]) # (220, 10)
    W = A.T
    
    # H can be random
    if version == 'rand_H':
        H = np.random.randn(k, N)
    
    # H can be chosen to minimize norm of A - WH
    elif version == 'op_H':
        epsilon = 1e-6
        for i in range(50):
            H = H * ((W.T @ X) / (W.T @ W @ H + epsilon))

    return W, H

def NNDSVD(X, k):
    
    # compute singuar vectors and values of X
    # U = (220, 10), V =(10, 256)
    U, S, V = svds(X, k)

    # initialize A and S  
    m, N = X.shape
    W = np.empty([m, k])
    H = np.empty([k, N])

    W[:, 1] = math.sqrt(S[1]) * U[:, 1]
    H[1, :] = math.sqrt(S[1]) * V[1, :]

    for j in range(k):
        
        x, y = U[:, j], V[j, :]
        xp, xn, yp, yn = pos(x), neg(x), pos(y), neg(y)
        xpnrm, ypnrm = np.linalg.norm(xp), np.linalg.norm(yp)
        mp = xpnrm * ypnrm
        xnnrm, ynnrm = np.linalg.norm(xn), np.linalg.norm(yn)
        mn = xnnrm * ynnrm

        if mp > mn:
            u = xp / xpnrm
            v = yp / ypnrm
            sigma = mp
        else:
            u = xn / xnnrm
            v = yn / ynnrm
            sigma = mn

        W[:, j] = math.sqrt(S[j] * sigma) * u
        H[j, :] = math.sqrt(S[j] * sigma) * v.T
    
    return W, H

def pos(X): 

    return np.where(X>0, X, 0)

def neg(X):
    
    return np.where(X<0, abs(X), 0)

if __name__ == "__main__":

    X = loadmat('Swimmer.mat')['X'].astype(float)
    k = 10
    kmeans(X, k)
    #W, H = NNDSVD(X.astype(float), k)
    #print(W, H)
