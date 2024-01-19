import numpy as np


# BFGS Algorithm
def bfgs(f, grad_f, x0, max_iter=100, epsilon=1e-5):
    n = len(x0)
    Bk_inv = np.eye(n)
    xk = np.array(x0)

    for _ in range(max_iter):
        # Calculate gradient
        gk = grad_f(xk)

        # Stopping condition
        if np.linalg.norm(gk) < epsilon:
            break

        # Search direction
        pk = -Bk_inv @ gk

        # Line search (using backtracking here, but other methods can be used)
        alpha = 1
        while f(xk + alpha * pk) > f(xk) + 0.1 * alpha * np.dot(gk, pk):
            alpha *= 0.5

        # Update x
        xk_new = xk + alpha * pk

        # Compute s_k and y_k
        sk = xk_new - xk
        yk = grad_f(xk_new) - gk
        rho_k = 1.0 / (yk @ sk)

        # Update Bk_inv using BFGS formula
        I = np.eye(n)
        Bk_inv = (I - rho_k * np.outer(sk, yk)) @ Bk_inv @ (
            I - rho_k * np.outer(yk, sk)
        ) + rho_k * np.outer(sk, sk)

        # Update for next iteration
        xk = xk_new

    return xk


# Define the Rosenbrock function
def rosenbrock(x, a=1, b=100):
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


# Define the gradient of the Rosenbrock function
def grad_rosenbrock(x, a=1, b=100):
    return np.array(
        [
            -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
            2 * b * (x[1] - x[0] ** 2),
        ]
    )


# Initial guess
x0 = np.array([0.0, 0.0])
# Test the BFGS algorithm on Rosenbrock function with different values of a (from 1 to 10)
bfgs_rosenbrock_results = {}
for a in range(1, 11):
    result = bfgs(lambda x: rosenbrock(x, a=a), lambda x: grad_rosenbrock(x, a=a), x0)
    bfgs_rosenbrock_results[a] = result

print(bfgs_rosenbrock_results)
