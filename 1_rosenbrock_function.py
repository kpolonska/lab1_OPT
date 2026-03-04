import numpy as np
import time

def calculate_objective(x1, x2):
    function = 100 * (x2 - x1**2)**2 + (1-x1)**2
    return function


def compute_gradient(x1, x2):
    df_dx1 = -400 * x1 * (x2 - x1**2) + 2*(x1 - 1)
    df_dx2 = 200 * (x2 - x1**2)

    return [df_dx1, df_dx2]

def gradient_descent(x01, x02, lamda, max_iter, stop_cr):
    x1 = x01
    x2 = x02

    start_time = time.time()
    prev_step = calculate_objective(x1, x2)

    for k in range(max_iter):
        g1, g2 = compute_gradient(x1, x2)

        x1_new = x1 - lamda * g1
        x2_new = x2 - lamda * g2

        f = calculate_objective(x1_new, x2_new)
        if abs(prev_step - f) < stop_cr:
            print(f'Found the min of f: {f}. The point [{x1_new}, {x2_new}]')
            print(f'Time taken: {time.time() - start_time}. Number of iterations: {k+1}.')
            return f

        x1 = x1_new
        x2 = x2_new
        prev_step = f

    print('Exceeded number of iterations. Exited function.')
    return 0


lamdas = [0.001]

for i in lamdas:
    gradient_des = gradient_descent(-2, 2, i, 10000, 0.00001)


