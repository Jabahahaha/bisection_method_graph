import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return x**3 - x - 2

# Derivative of the function
def df(x):
    return 3*x**2 - 1

# Bisection Method
def bisection(f, a, b, tol=1e-6, max_iter=100):
    iterations = []
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints a and b.")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        iterations.append(c)
        
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            break
        
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    return c, iterations

# Fixed Point Iteration
def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    xk = x0
    iterations = [xk]
    
    for _ in range(max_iter):
        xk_next = g(xk)
        iterations.append(xk_next)
        
        if abs(xk_next - xk) < tol:
            break
        
        xk = xk_next
    
    return xk, iterations

# Transformation for Fixed Point Iteration
alpha = 0.01
def g(x):
    return x - alpha * f(x)

# Real solution for comparison
real_solution = 1.5213797068045678

# Run Bisection Method
root_bisection, iterations_bisection = bisection(f, 1, 2)

# Run Fixed Point Iteration
root_fixed_point, iterations_fixed_point = fixed_point_iteration(g, 1.5)

# Function value at current approximation
f_values_fp = [f(x) for x in iterations_fixed_point]
f_values_bisect = [f(x) for x in iterations_bisection]

# Distance between successive iterations
step_sizes_fp = [abs(iterations_fixed_point[i] - iterations_fixed_point[i-1]) for i in range(1, len(iterations_fixed_point))]
step_sizes_bisect = [abs(iterations_bisection[i] - iterations_bisection[i-1]) for i in range(1, len(iterations_bisection))]

# Distance to real solution
distances_to_solution_fp = [abs(x - real_solution) for x in iterations_fixed_point]
distances_to_solution_bisect = [abs(x - real_solution) for x in iterations_bisection]

# Plot Function Value Convergence
plt.figure(figsize=(10, 6))
plt.plot(f_values_fp, label='Fixed Point')
plt.plot(f_values_bisect, label='Bisection')
plt.xlabel('Iteration Number')
plt.ylabel('Function Value |f(x_k)|')
plt.title('Convergence of Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot Step Size Convergence
plt.figure(figsize=(10, 6))
plt.plot(step_sizes_fp, label='Fixed Point')
plt.plot(step_sizes_bisect, label='Bisection')
plt.xlabel('Iteration Number')
plt.ylabel('Step Size |x_k - x_(k-1)|')
plt.title('Convergence of Step Size')
plt.legend()
plt.grid(True)
plt.show()

# Plot Distance to Real Solution
plt.figure(figsize=(10, 6))
plt.plot(distances_to_solution_fp, label='Fixed Point')
plt.plot(distances_to_solution_bisect, label='Bisection')
plt.xlabel('Iteration Number')
plt.ylabel('Distance to Real Solution |x_k - x*|')
plt.title('Convergence to Real Solution')
plt.legend()
plt.grid(True)
plt.show()
