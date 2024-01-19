import numpy as np


# DFP Algorithm
def dfp(f, grad_f, x0, max_iter=100, epsilon=1e-5):
    n = len(x0)
    Hk = np.eye(n)
    xk = np.array(x0)

    for _ in range(max_iter):
        # Calculate gradient
        gk = grad_f(xk)

        # Stopping condition
        if np.linalg.norm(gk) < epsilon:
            break

        # Search direction
        pk = -np.dot(Hk, gk)

        # Line search (using backtracking here)
        alpha = 1
        while f(xk + alpha * pk) > f(xk) + 0.1 * alpha * np.dot(gk, pk):
            alpha *= 0.5

        # Update x
        xk_new = xk + alpha * pk

        # Update Hk
        sk = xk_new - xk
        yk = grad_f(xk_new) - gk
        rho = 1.0 / (yk @ sk)
        Hk_new = (np.eye(n) - rho * np.outer(sk, yk)) @ Hk @ (
            np.eye(n) - rho * np.outer(yk, sk)
        ) + rho * np.outer(sk, sk)

        # Update for next iteration
        xk = xk_new
        Hk = Hk_new

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

rosenbrock_results = {}
for a in range(1, 11):
    result = dfp(lambda x: rosenbrock(x, a=a), lambda x: grad_rosenbrock(x, a=a), x0)
    rosenbrock_results[a] = result

print(rosenbrock_results)
