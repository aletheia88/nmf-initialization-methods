import numpy as np
from NMF_basic import *

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
