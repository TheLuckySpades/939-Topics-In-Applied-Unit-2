import numpy as np
import matplotlib.pyplot as plt

#Solution is 3,4
#A = np.array([[2,1],[1,2]])
A = np.array([[1000, 1],
              [1, 1]])
x_0 = np.array([2,2])
b = np.array([10,11])
r = b - (A @ x_0)

residual_norms = []

#Also called gradient descent
def steepestDescent(A,x_0,r,residual_norms):
    iters = 0
    while(iters < 100):
        residual_norms.append(np.linalg.norm(r))
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ A @ r)
        x_0 = x_0 + (alpha * r)
        r = r - alpha * A @ r
        iters += 1
        #print(np.linalg.norm(r))

    residual_norms.append(np.linalg.norm(r))
    print(x_0)

steepestDescent(A,x_0,r,residual_norms)

#Precondition via diagonalizing

#A = np.array([[2,1],[1,2]])
A = np.array([[1000, 1],
              [1, 1]])
x_0 = np.array([2,2])
b = np.array([10,11])
r = b - (A @ x_0)

D_inv = 1.0 / np.diag(A)

z = D_inv * r
normstwo = []

steepestDescent(A,x_0,z,normstwo)

plt.semilogy(residual_norms, marker='o', label="Steepest Descent")
plt.semilogy(normstwo, marker='s', label="Preconditioned")
plt.xlabel("Iteration")
plt.ylabel(r"$\|r_k\|_2$")
plt.title("Steepest Descent Convergence")
plt.grid(True)
plt.show()