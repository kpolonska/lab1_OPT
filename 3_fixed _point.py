import numpy as np
import time

def calculate_objective(x, f):
    if f == 1:
        g1 = (1/2) * (x - np.log((1 + x)))**2
        return g1
    if f == 2:
        g2 = (1/2) * (x - np.log((2 + x)))**2
        return g2
    else:
        print('Error, f not correct.')
        return 0


def compute_gradient(x, f):
    if f == 1:
        d1 = (x - np.log(1 + x)) * (1 - (1 / (1 + x)))
        return d1
    if f == 2:
        d2 = (x - np.log(2 + x)) * (1 - (1 / (2 + x)))
        return d2
    else:
        print('Error, f not correct.')
        return 0

def gradient_descent(x, lamda, max_iter, stop_cr, func_number):
    x1 = x

    start_time = time.time()
    prev_step = calculate_objective(x, func_number)

    for k in range(max_iter):

        gradient = compute_gradient(x1, func_number)
        x1_new = x1 - lamda * gradient
        f = calculate_objective(x1_new, func_number)

        if k % 10 == 0:
            print(f'On iteration {k}: value f = {f}, x = {x1_new}')

        if abs(prev_step - f) < stop_cr:
            print(f'Found the min of f: {f}. The point [{x1_new}]')
            print(f'Time taken: {time.time() - start_time}. Number of iterations: {k+1}.')
            return f

        x1 = x1_new
        prev_step = f

    print('Exceeded number of iterations. Exited function.')
    return 0


x = 2 # because the region is [0, 2] and we take the max
L1 = (x - np.log(1 + x)) * (1 - (1 / (1 + x))) # online calc
L2 = (x - np.log(2 + x)) * (1 - (1 / (2 + x)))  # online calc

result_g1 = gradient_descent(0.1, 1/L1, 100, 0.00000001, 1)
result_g2 = gradient_descent(0.1, 1/L2, 100, 0.00000001, 2)

