import numpy as np
import time
import matplotlib.pyplot as plt

def calculate_objective(x, A, b):
    m = A.shape[0]
    mn = A @ x - b
    function = (1 / (2*m)) * (mn @ mn)
    return function


def compute_gradient(x, A, b):
    m = A.shape[0]
    return (1/m) * (A.T @ (A@x - b))


def gradient_descent(x0, A, b, lamda, max_iter, stop_cr):
    x = x0.copy()

    start_time = time.time()
    objective_values = []

    for k in range(max_iter):
        g = compute_gradient(x, A, b)

        x1_new = x - lamda * g

        f = calculate_objective(x1_new, A, b)
        objective_values.append(f)
        if np.linalg.norm(g) < stop_cr:
            print(f'Found the min of f: {f}. The point x=[{x1_new}]')
            print(f'Time taken: {time.time() - start_time}. Number of iterations: {k+1}.')
            return objective_values

        x = x1_new

    print('Exceeded number of iterations. Exited function.')
    return objective_values

A = np.array([[2, 3, 4], [3, 5, 1], [5, 6, 7]], dtype=float)
b = np.array([1, 8, 9], dtype=float)

beta = 1/3 * (np.linalg.norm(A, 2))**2
x0 = np.zeros(A.shape[1])

one = np.linalg.norm(A.T @ A, 2)
two = np.linalg.norm(A.T @ b)

L = (1/3) * (one * 20 + two)
lamdas = [0.1, 1/beta, 1/L]
labels = ['λ = 0.1', 'λ = 1/β', 'λ = 1/L']

results = []
for i in lamdas:
    gradient_des = gradient_descent(x0, A, b, i, 50, 0.001)
    results.append(gradient_des)


for i, (result, label) in enumerate(zip(results, labels)):
    plt.plot(result, label=label)

plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()


