# Explores Condition Numbers and Convergence of different iteration schemes.
# Nick White Math 939, 2/2/2026
# 

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Need to figure out how to generate a Diagonally Dominant Matrix with Prescribed spectrum



def random_diag_dominant_prescribed_spectrum(lam, eps=0.05, seed=0):
    n = len(lam)
    rng = np.random.default_rng(seed)

    # random skew-symmetric matrix
    S = rng.standard_normal((n, n))
    S = S - S.T

    Q = expm(eps * S)  # e^S is orthogonal. Then e^epsS is orthogonal as epsS is still skew symmetric
    A = Q @ np.diag(lam) @ Q.T
    return A

def is_diagonally_dominant(A):
    return np.all(np.abs(np.diag(A)) >= np.sum(np.abs(A), axis=1) - np.abs(np.diag(A)))


def Jacobi_Iter(A, x0, xtrue, b, maxIters):

    # Jacobi iteration
    D = np.diag(A) # extract diagonals
    R = A-np.diag(D) # A with diagonals = 0
    err = np.zeros(maxIters+1)
    res = np.zeros(maxIters+1)
    x = x0
    err[0] = np.linalg.norm(x-x_true)
    res[0] = np.linalg.norm(b-A@x)

    for k in range(maxIters):
        x = (b-R@x)/D # The main Jacobi step
        err[k+1] = np.linalg.norm(x-x_true)
        res[k+1] = np.linalg.norm(b-A@x)

#     # plot results
#     k = np.arange(iters+1)
#     plt.semilogy(k,err,'-o',markevery=5,label='abs error')
#     plt.semilogy(k,res,'--s',markevery=5,label='abs residual')
#     plt.xlabel('iteration count')
#     plt.legend()
#     plt.grid(True,which='both',linestyle=':')
#     plt.tight_layout()
#     plt.savefig('jacobi_convergence.png',dpi=300,bbox_inches='tight')
#     plt.show()
    return err, res

n = 5
k = 5
rng = np.random.default_rng(0)
lambda_vals = np.linspace(1.0, 10, n)
lam_n = np.logspace(-3, 2, k)
x_true = rng.standard_normal(n)
maxIters = 50
x0 = np.zeros(n)
errs = np.zeros((maxIters+1, k))
ress = np.zeros((maxIters+1, k))
i = 0
k = np.arange(maxIters + 1)
for lam in lam_n:
    lambda_vals[-1] = lam
    A = random_diag_dominant_prescribed_spectrum(lambda_vals,1e-5)
    print(A)
    if not is_diagonally_dominant(A):
        print('ERROR - A not diagonally dominant, might need to make eps smaller')
        break
    b = A@x_true
    errs[:,i], ress[:, i] = Jacobi_Iter(A, x0, x_true, b, maxIters)
    plt.semilogy(k, errs[:, i], '-o', markevery=5, label=f'absolute err, condition number ={lam}')
    plt.semilogy(k, ress[:, i], '--s', markevery=5, label=f'relative err, condition number ={lam}')
    plt.xlabel('Iteration Count')
    plt.legend()
    plt.grid(True,which='both', linestyle=':')
    i = i +1
plt.show()
    

