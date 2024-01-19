import numpy as np


# Fletcher-Reeves (FR) Conjugate Gradient Algorithm
def fr_conjugate_gradient(f, grad_f, x0, max_iter=1000, epsilon=1e-5):
    xk = np.array(x0)
    gk = grad_f(xk)
    pk = -gk
    for _ in range(max_iter):
        # Line search to find alpha_k
        alpha_k = line_search(f, grad_f, xk, pk)

        # Update xk
        xk_new = xk + alpha_k * pk

        # Calculate new gradient
        gk_new = grad_f(xk_new)

        # Check convergence
        if np.linalg.norm(gk_new) < epsilon:
            break

        # Fletcher-Reeves update
        beta_k = np.dot(gk_new, gk_new) / np.dot(gk, gk)

        # Update pk
        pk = -gk_new + beta_k * pk

        # Update for next iteration
        xk = xk_new
        gk = gk_new

    return xk


# Line search function to find an appropriate alpha using backtracking
def line_search(f, grad_f, xk, pk, alpha=1, rho=0.5, c=1e-4):
    while f(xk + alpha * pk) > f(xk) + c * alpha * np.dot(grad_f(xk), pk):
        alpha *= rho
    return alpha


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
x0 = np.array([-1.2, 1.0])
# Test the FR conjugate gradient algorithm on Rosenbrock function with different values of a (from 1 to 10)

fr_rosenbrock_results = {}
for a in range(1, 11):
    result = fr_conjugate_gradient(
        lambda x: rosenbrock(x, a=a), lambda x: grad_rosenbrock(x, a=a), x0
    )
    fr_rosenbrock_results[a] = result

print(fr_rosenbrock_results)
