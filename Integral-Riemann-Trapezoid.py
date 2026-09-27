import numpy as np
from scipy.integrate import trapezoid
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Defining our variables 

k = 8.9875e9 # Coulomb's constant 
q1 = 1.0e-6  # charge 1 in Coulombs 
q2 = 1.0e-6 #charge 2 in Coulombs 
R = 0.01 # lower bound (starting seperation), meters 
r_max = 10.0 # truncated stand-in for infinity, meters 
n_points = 5000 

# Analytical Solution 

def integrand (r, q1, q2, k):
    return (k * q1 * q2) / (r**2)

def W_analytic(R, q1, q2, k):
    return (k * q1 * q2) / R 

# Simple Rienmann Sum
    # chop up R to r_max into a bunch of equal sides rectangles (n points), 
    # evaluate at the left edge og each then multiply by the width and add together

def riemann_sum(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points  + 1)  # edge of each rectangle
    dx = x[1] - x[0]                      # width of each rectangle 

    total = 0.0

    for i in range(n_points):
        total += f(x[i], *f_args) * dx  # height (edge) * width 
    return total

# compare analytical solution to riemann 

W_re_numeric = riemann_sum(integrand, R, r_max, n_points, q1, q2, k)
W_true = W_analytic(R, q1, q2, k)

print(f"Riemann sum estimate: {W_re_numeric:.6e} J")
print(f"Analytic value:       {W_true:.6e} J")
print(f"Relative error:       {abs(W_re_numeric - W_true)/W_true:.3e}")

# Trapezoidal Rule
    # similar to the Rienmann but uses an adverage of the left and right edge height in place of just the left edge

def trapezoidal_sum(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points  + 1)
    dx = x[1] - x[0] 

    total = 0.0

    for i in range(n_points):
        height = (f(x[i], *f_args) + f(x[i + 1], *f_args)) / 2 # averages the two function values (heights) — it evaluates f twice
        total += height * dx
    return total

# compare analytical solution to trapezoid

W_tr_numeric = trapezoidal_sum(integrand, R, r_max, n_points, q1, q2, k)
W_true = W_analytic(R, q1, q2, k)

print(f"Trapezoidal sum estimate: {W_tr_numeric:.6e} J")
print(f"Analytic value:       {W_true:.6e} J")
print(f"Relative error:       {abs(W_tr_numeric - W_true)/W_true:.3e}")
