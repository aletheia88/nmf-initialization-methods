import numpy as np
from nmf import *
from tqdm import tqdm
from scipy.io import loadmat
import json

def get_optimal_factorization(eval_iters, X, k, nmf_iters, method):

    m, N = X.shape
    
    # initialize A, S
    op_A = np.zeros(m, k)
    op_S = np.zeros(k, N)
    op_loss = float('inf')

    for i in range(eval_iters):

        A, S, loss = multiplicative_update(X, k, nmf_iters, method)
        
        if loss < op_loss:
            op_A, op_S = A, S

    return op_A, op_S, op_loss

def get_nmf_loss_stats(eval_iters, X, k, nmf_iters, method): 
    
    losses = []

    for i in tqdm(range(eval_iters)):
        losses.append(multiplicative_update(X, k, nmf_iters, method)[-1])

    return losses, np.mean(losses), np.median(losses) 

def get_nmf_iters_stats(X, k, target_loss, method):

    nmf_iters = []

    for i in tqdm(range(100)):

        nmf_iters.append(get_nmf_iters_given_target_loss(X, k, target_loss,
                                                                method))
    return nmf_iters, np.mean(nmf_iters), np.median(nmf_iters)

def get_nmf_iters_given_target_loss(X, k, target_loss, method):

    nmf_iters = 1
    nmf_loss = 1

    while nmf_loss > target_loss:
        
        if nmf_iters > 1000: break

        _, _, nmf_loss = multiplicative_update(X, k, nmf_iters, method)

        if nmf_loss > target_loss:
            nmf_iters += 1
    
    return nmf_iters

def main():
    X = loadmat('Swimmer.mat')['X'].astype(float)
    k = 10
    eval_iters = 100
    nmf_iters = 100
    target_loss = 0.5
    '''
    methods = ['gaussian', 'uniform', 'laplacian', 
                'poisson', 'kmeans', 'nndsvd']
    '''
    methods = ['kmeans', 'nndsvd'] 
    results = {}
    
    for method in methods:
        
        _, mean_nmf_loss, median_nmf_loss = get_nmf_loss_stats(eval_iters, X, k, 
                                                        nmf_iters, method)
        
        _, mean_nmf_iter, median_nmf_iter = get_nmf_iters_stats(X, k, 
                                                        target_loss, method)
        results[method] = {
                        'mean_nmf_loss':mean_nmf_loss,
                        'median_nmf_loss':median_nmf_loss,
                        'mean_nmf_iter':mean_nmf_iter,
                        'median_nmf_iter':median_nmf_iter 
                        }

        print(f'{method}\n{results[method]}')
        
    f = open('evaluation_results.json')
    json.dump(results, f, indent=2)

if __name__ == "__main__":

    main()        
