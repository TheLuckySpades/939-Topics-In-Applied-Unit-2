import numpy as np
import matplotlib.pyplot as plt
import copy

global maxIters
maxIters = 5000

def spd_prescribed_spectrum(n,kappa=1e6,seed=0):
    rng=np.random.default_rng(seed)
    Q,_=np.linalg.qr(rng.standard_normal((n,n)))
    lam=np.geomspace(1.0,kappa,n)
    A=Q@np.diag(lam)@Q.T
    return A




#Solution is 3,4
#A = np.array([[2,1],[1,2]])
Aprime = spd_prescribed_spectrum(10)
A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
r = b - (A @ x_0)

residual_norms = []

#Also called gradient descent
def steepestDescent(A,x_0,r,residual_norms):
    iters = 0
    while((iters < maxIters) and (np.linalg.norm(r) > 10**-15)):
        residual_norms.append(np.linalg.norm(r))
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ A @ r)
        x_0 = x_0 + (alpha * r)
        r = r - alpha * A @ r
        iters += 1
        #print(np.linalg.norm(r))

    residual_norms.append(np.linalg.norm(r))
    print("GD done:",x_0)

steepestDescent(A,x_0,r,residual_norms)

#Precondition via diagonalizing

def preconditionedSteepestDescent(A, x, b, residual_norms):
    r = b - A @ x
    D_inv = 1.0 / np.diag(A)
    iters = 0
    print(np.linalg.norm(r))
    
    while((iters < maxIters) and (np.linalg.norm(r) > 10**-15)):
        residual_norms.append(np.linalg.norm(r))

        z = D_inv * r
        alpha = (r @ z) / (z @ (A @ z))   

        x = x + alpha * z
        r = r - alpha * A @ z
        iters += 1
    print("PCG Done:",x)


def conjugateGradient(A, x0, b,residual_norms):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()

    residual_norms.append(np.linalg.norm(r))
    iters = 0
    print(x)
    while(iters < maxIters and (np.linalg.norm(r) > 10**-15)):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)

        x = x + alpha * p
        r_new = r - alpha * Ap

        residual_norms.append(np.linalg.norm(r_new))

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        iters += 1
    print("CG Done:",x)


def preconditionedConjugateGradient(A, x0, b, residual_norms):
    x = x0.copy()
    r = b - A @ x

    D_inv = 1.0 / np.diag(A)
    z = D_inv * r
    p = z.copy()

    rz_old = r @ z
    residual_norms.append(np.linalg.norm(r))

    iters = 0
    while iters < maxIters and np.linalg.norm(r) > 1e-12:
        Ap = A @ p
        alpha = rz_old / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        residual_norms.append(np.linalg.norm(r))

        z = D_inv * r
        rz_new = r @ z

        beta = rz_new / rz_old
        p = z + beta * p

        rz_old = rz_new
        iters += 1

    print("PCG Done:", x)

#No don't
def randrand(A, x0, b, residual_norms):
    m, n = A.shape
    x = x0.copy()

    iters = 0
    r = b - A @ x
    res_norm = np.linalg.norm(r)

    while iters < 500 and res_norm > 1e-12:
        residual_norms.append(res_norm)

        # random row and column
        i = np.random.randint(m)
        j = np.random.randint(n)

        a_ij = A[i, j]
        if abs(a_ij) > 1e-14:
            # rank-1 update
            x[j] += r[i] / a_ij

        # update residual explicitly
        r = b - A @ x
        res_norm = np.linalg.norm(r)

        iters += 1

    print("RandRAND iterations:", iters)
    print("RandRAND solution:", x)





#A = np.array([[2,1],[1,2]])
A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
r = b - (A @ x_0)

normsTwo = []
normsThree = []
normsFour = []
normsFive = []

preconditionedSteepestDescent(A,x_0,b,normsTwo)

#A = np.array([[2,1],[1,2]])
A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
r = b - (A @ x_0)

conjugateGradient(A,x_0,b,normsThree)

#A = np.array([[2,1],[1,2]])
A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
r = b - (A @ x_0)

preconditionedConjugateGradient(A,x_0,b,normsFour)


A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2], dtype=float)
b = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
r = b - (A @ x_0)

np.random.seed(1)

#randrand(A,x_0,b,normsFive)

A = copy.deepcopy(Aprime)
x_0 = np.array([2,2,2,2,2,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
r = b - (A @ x_0)

eigvals = np.linalg.eigvalsh(A)
cond_A = eigvals.max() / eigvals.min()
print("Raw condition number:",cond_A)


D = np.diag(np.diag(A))
#A_tilde = np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))
A_tilde = np.diag(1/np.diag(A)) @ A
eigvals_tilde = np.linalg.eigvalsh(A_tilde)
cond_pre = eigvals_tilde.max() / eigvals_tilde.min()
print("Pre-conditioned condition number:",cond_pre)

print("Condition number change:",cond_A-cond_pre)

print(eigvals[-1])
print(eigvals_tilde[-1])

plt.plot(eigvals, '.', label='Eigenvalues of A')
plt.plot(eigvals_tilde, 'x', label='Eigenvalues of Jacobi-preconditioned')
plt.legend()
plt.title("Eigenvalue distributions")
plt.grid(True)
plt.show()




plt.semilogy(residual_norms, marker='.', label="GD")
plt.semilogy(normsTwo, marker='.', label="PGD")
plt.semilogy(normsThree, marker='.', label="CG")
plt.semilogy(normsFour, marker='.', label="PCG")
plt.legend()
#plt.semilogy(normsFive, marker='p', label="RandRAND")
plt.xlabel("Iteration")
plt.ylabel(r"$\|r_k\|_2$")
plt.title("Convergence Comparisons")
plt.grid(True)
plt.show()